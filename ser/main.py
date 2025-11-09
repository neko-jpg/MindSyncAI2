import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig, OmegaConf, ListConfig
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

from .data_utils import build_dataset
from data import (
    RavdessDataset,
    CremaDDataset,
    TessDataset,
    SaveeDataset,
    EmoDBDataset,
    CombinedSERDataset,
)
from .losses import build_loss, SupConLoss
from .models.hybrid_ser import HybridSERNet
from .models.mobile_crnn_v1 import MobileCRNNv1
from .models.wav2vec_hybrid import Wav2Vec2SERNet
from .runtime.environment import EnvFeatureExtractor
from .runtime.ood import GScoreEstimator
from .runtime.calibration import DirichletCalibrator, IsotonicBackoff, CalibrationArtifactBundle
from .runtime.signatures import SignaturePackage, SignatureWriter
from .training.adversarial import SpectralPGDAttacker
from .training.augmentations import BatchAugmentor, FeatureAugmentor, TTAAugmentor, OpenSetBatchMixer
from .training.callbacks import EarlyStopping
from .training.distillation import DistillationHelper
from .training.ema import build_ema

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - older sklearn fallback
    StratifiedGroupKFold = None
from sklearn.model_selection import StratifiedKFold


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)
    bins = torch.linspace(0.0, 1.0, num_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    for i in range(num_bins):
        in_bin = (confidences > bins[i]) * (confidences <= bins[i + 1])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece.item())


def collate_batch(batch):
    features = [item["features"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    speaker_ids = torch.tensor([item["speaker_id"] for item in batch], dtype=torch.long)

    feature_lengths = torch.tensor([feat.shape[-1] for feat in features], dtype=torch.long)
    max_feat_len = int(feature_lengths.max())
    padded_features = []
    for feat in features:
        pad_amount = max_feat_len - feat.shape[-1]
        if pad_amount > 0:
            feat = F.pad(feat, (0, pad_amount))
        padded_features.append(feat)
    feature_tensor = torch.stack(padded_features)

    if "waveform" in batch[0]:
        waveforms = [item["waveform"] for item in batch]
        waveform_lengths = torch.tensor([item["waveform_length"] for item in batch], dtype=torch.long)
        max_wave_len = int(waveform_lengths.max())
        waveform_tensor = torch.zeros(len(batch), max_wave_len, dtype=waveforms[0].dtype)
        for idx, waveform in enumerate(waveforms):
            waveform_tensor[idx, : waveform.shape[-1]] = waveform
    else:
        waveform_tensor = None
        waveform_lengths = None

    return {
        "features": feature_tensor,
        "label": labels,
        "speaker_id": speaker_ids,
        "length": feature_lengths,
        "waveform": waveform_tensor,
        "waveform_length": waveform_lengths,
    }


def compute_class_counts(dataset, indices: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for idx in indices:
        label = dataset.samples[idx]["label"]
        counts[label] += 1
    return counts


def build_training_sampler(cfg: DictConfig, dataset, train_indices: Sequence[int]):
    sampling_cfg = cfg.dataset.get('sampling') or {}
    strategy = str(sampling_cfg.get('strategy', 'proportional')).lower()
    allowed_strategies = {'proportional', 'balanced', 'uniform'}
    if strategy not in allowed_strategies:
        raise ValueError(f"Unsupported dataset.sampling.strategy='{strategy}'. Expected one of {allowed_strategies}.")

    dataset_weights_cfg = getattr(dataset, 'dataset_weights', {}) or {}
    dataset_weights = {str(k).lower(): float(v) for k, v in dataset_weights_cfg.items()}
    has_custom_weight = any(abs(w - 1.0) > 1e-8 for w in dataset_weights.values())

    if strategy == 'proportional' and not has_custom_weight:
        return None

    train_indices = list(train_indices)
    if len(train_indices) == 0:
        return None

    dataset_labels: List[str] = []
    for idx in train_indices:
        meta = dataset.samples[idx]
        ds_name = str(meta.get('dataset', 'unknown')).lower()
        dataset_labels.append(ds_name)

    counts = Counter(dataset_labels)
    total = len(train_indices)
    sample_weights: List[float] = []
    for ds_name in dataset_labels:
        if strategy == 'balanced':
            base = 1.0 / max(counts[ds_name], 1)
        elif strategy == 'uniform':
            base = 1.0
        else:  # proportional
            base = counts[ds_name] / total
        custom = dataset_weights.get(ds_name, dataset_weights.get(ds_name.upper(), dataset_weights.get(ds_name.title(), 1.0)))
        base *= custom
        sample_weights.append(base)

    if not any(weight > 0 for weight in sample_weights):
        return None

    weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
    if torch.allclose(weights_tensor, torch.full_like(weights_tensor, weights_tensor[0])):
        if strategy == 'proportional' and not has_custom_weight:
            return None

    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(sample_weights), replacement=True)
    descriptor = strategy
    if has_custom_weight:
        descriptor += ' + custom weights'
    return sampler, descriptor

def _ensure_list_config(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_source_cfg(dataset_cfg: DictConfig, name: str):
    sources = dataset_cfg.get("sources")
    if sources is not None and name in sources:
        return sources[name]
    return dataset_cfg.get(name)


def _dataset_root_from_cfg(default_root: str, source_dict: Dict[str, object]) -> str:
    candidate = source_dict.get("root") or source_dict.get("path") or default_root
    return hydra.utils.to_absolute_path(str(candidate))


def create_dataset_instance(
    name: str,
    dataset_root: str,
    cfg: DictConfig,
    segment_duration: Optional[float],
    hop_duration: Optional[float],
    min_coverage: float,
):
    name = name.lower()
    if name == "ravdess":
        return RavdessDataset(
            data_dir=dataset_root,
            sample_rate=cfg.features.sample_rate,
            n_mels=cfg.features.n_mels,
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            min_coverage=min_coverage,
        )
    if name in {"cremad", "crema_d"}:
        return CremaDDataset(
            data_dir=dataset_root,
            sample_rate=cfg.features.sample_rate,
            n_mels=cfg.features.n_mels,
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            min_coverage=min_coverage,
        )
    if name == "tess":
        return TessDataset(
            data_dir=dataset_root,
            sample_rate=cfg.features.sample_rate,
            n_mels=cfg.features.n_mels,
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            min_coverage=min_coverage,
        )
    if name == "savee":
        return SaveeDataset(
            data_dir=dataset_root,
            sample_rate=cfg.features.sample_rate,
            n_mels=cfg.features.n_mels,
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            min_coverage=min_coverage,
        )
    if name in {"emodb", "emo_db", "berlin"}:
        return EmoDBDataset(
            data_dir=dataset_root,
            sample_rate=cfg.features.sample_rate,
            n_mels=cfg.features.n_mels,
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            min_coverage=min_coverage,
        )
    raise ValueError(f"Unsupported dataset name: {name}")


def build_dataset(cfg: DictConfig):
    dataset_cfg = cfg.dataset
    dataset_names = _ensure_list_config(dataset_cfg.get("names"))
    if not dataset_names:
        dataset_names = _ensure_list_config(dataset_cfg.get("name"))
    if not dataset_names:
        raise ValueError("dataset.name or dataset.names must be specified in the configuration.")

    segment_default = dataset_cfg.get("segment_duration_s", None)
    hop_default = dataset_cfg.get("hop_duration_s", None)
    min_cov_default = dataset_cfg.get("min_segment_coverage", 0.6)
    default_root = dataset_cfg.get("root") or dataset_cfg.get("path") or cfg.data_dir

    datasets = []
    for raw_name in dataset_names:
        name = str(raw_name).lower()
        source_cfg = _resolve_source_cfg(dataset_cfg, name) or {}
        if isinstance(source_cfg, DictConfig):
            source_dict = OmegaConf.to_container(source_cfg, resolve=True)
        elif isinstance(source_cfg, dict):
            source_dict = dict(source_cfg)
        else:
            source_dict = {}

        dataset_root = _dataset_root_from_cfg(default_root, source_dict)
        segment_duration = source_dict.get("segment_duration_s", segment_default)
        hop_duration = source_dict.get("hop_duration_s", hop_default)
        min_coverage = source_dict.get("min_segment_coverage", min_cov_default)

        dataset = create_dataset_instance(
            name=name,
            dataset_root=dataset_root,
            cfg=cfg,
            segment_duration=segment_duration,
            hop_duration=hop_duration,
            min_coverage=min_coverage if min_coverage is not None else min_cov_default,
        )
        datasets.append(dataset)

    if len(datasets) == 1:
        return datasets[0]
    return CombinedSERDataset(datasets)


def build_model(cfg, device: torch.device):
    name = cfg.model.name.lower()
    if name == "hybrid_ser":
        model = HybridSERNet(cfg)
    elif name == "mobile_crnn":
        model = MobileCRNNv1(num_classes=cfg.dataset.num_classes)
    elif name == "wav2vec2_hybrid":
        model = Wav2Vec2SERNet(cfg)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    return model.to(device)


def create_optimizer(cfg, model):
    optimizer_cfg = cfg.training.optimizer
    name = optimizer_cfg.name.lower()
    if name == "adamw":
        return AdamW(
            model.parameters(),
            lr=optimizer_cfg.lr,
            weight_decay=optimizer_cfg.weight_decay,
            betas=tuple(optimizer_cfg.betas),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_cfg.name}")


def create_scheduler(cfg, optimizer):
    sched_cfg = cfg.training.scheduler
    warmup_epochs = sched_cfg.warmup_epochs
    schedulers = []
    milestones = []
    if warmup_epochs and warmup_epochs > 0:
        schedulers.append(LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs))
        milestones.append(warmup_epochs)
    schedulers.append(
        CosineAnnealingLR(
            optimizer,
            T_max=max(1, sched_cfg.t_max - warmup_epochs),
            eta_min=sched_cfg.eta_min,
        )
    )
    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def refresh_batch_norm(model, data_loader: DataLoader, device: torch.device) -> None:
    model.train()
    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device, non_blocking=True)
            waveforms = batch.get("waveform")
            if waveforms is not None:
                waveforms = waveforms.to(device, non_blocking=True)
                waveform_lengths = batch["waveform_length"].to(device, non_blocking=True)
            else:
                waveform_lengths = None
            lengths = batch["length"].to(device, non_blocking=True)
            model(
                features=features,
                waveforms=waveforms,
                waveform_lengths=waveform_lengths,
                lengths=lengths,
            )
    model.eval()


def evaluate(
    model,
    data_loader: DataLoader,
    criterion,
    device: torch.device,
    tta: Optional[TTAAugmentor] = None,
    temperature: Optional[torch.Tensor] = None,
    collect_logits: bool = False,
    collect_embeddings: bool = False,
    amp_enabled: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    env_extractor: Optional[EnvFeatureExtractor] = None,
) -> Dict[str, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    preds, labels = [], []
    logits_list = []
    embedding_list = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device, non_blocking=True)
            targets = batch["label"].to(device, non_blocking=True)
            lengths = batch["length"].to(device, non_blocking=True)
            waveforms = batch.get("waveform")
            waveform_lengths = batch.get("waveform_length")
            if waveforms is not None:
                waveforms = waveforms.to(device, non_blocking=True)
            if waveform_lengths is not None:
                waveform_lengths = waveform_lengths.to(device, non_blocking=True)
            env_features = None
            if env_extractor is not None:
                env_features, _ = env_extractor(waveforms, waveform_lengths, device=device)

            autocast_kwargs = {"enabled": amp_enabled and device.type == "cuda"}
            if autocast_kwargs["enabled"] and amp_dtype is not None:
                autocast_kwargs["dtype"] = amp_dtype
            with autocast(**autocast_kwargs):
                if tta is not None:
                    outputs = tta(
                        model,
                        features,
                        waveforms,
                        waveform_lengths,
                        lengths=lengths,
                        env_features=env_features,
                    )
                else:
                    outputs = model(
                        features=features,
                        waveforms=waveforms,
                        waveform_lengths=waveform_lengths,
                        lengths=lengths,
                        env_features=env_features,
                    )
                if temperature is not None:
                    outputs = outputs / temperature
                loss = criterion(outputs, targets)
            total_loss += loss.item()

            outputs_float = outputs.detach().float()
            probabilities = torch.softmax(outputs_float, dim=-1)
            pred = torch.argmax(probabilities, dim=-1)
            preds.append(pred.cpu().numpy())
            labels.append(targets.cpu().numpy())

            if collect_logits:
                logits_list.append(outputs_float.cpu())
            if collect_embeddings and hasattr(model, "latest_embedding"):
                embedding_list.append(model.latest_embedding.detach().cpu())

    preds = np.concatenate(preds)
    labels_np = np.concatenate(labels)
    avg_loss = total_loss / max(1, len(data_loader))
    uar = recall_score(labels_np, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels_np, preds, average="macro", zero_division=0)

    result = {"loss": avg_loss, "uar": uar, "macro_f1": macro_f1}
    if collect_logits and logits_list:
        result["logits"] = torch.cat(logits_list)
        result["labels"] = torch.from_numpy(labels_np)
    if collect_embeddings and embedding_list:
        result["embeddings"] = torch.cat(embedding_list)
    return result


def tune_temperature(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    device = logits.device
    log_temp = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = log_temp.exp().clamp(min=1e-3)
        loss = F.cross_entropy(logits / temperature, labels.to(device))
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temp.detach().exp().clamp(min=1e-3)


def maybe_build_distiller(cfg, device: torch.device):
    distill_cfg = cfg.training.distillation
    if not distill_cfg.enabled or not distill_cfg.teacher_checkpoint:
        return None

    checkpoints = distill_cfg.teacher_checkpoint
    if isinstance(checkpoints, (list, tuple)):
        ckpt_list = list(checkpoints)
    else:
        ckpt_list = [checkpoints]
    teachers = []
    for ckpt in ckpt_list:
        teacher = build_model(cfg, device)
        ckpt_path = hydra.utils.to_absolute_path(ckpt)
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            teacher.load_state_dict(state["state_dict"])
        else:
            teacher.load_state_dict(state)
        teacher.eval()
        teachers.append(teacher)
        print(f"[Distillation] Teacher loaded from {ckpt_path}")
    return DistillationHelper(
        teacher_model=teachers,
        temperature=distill_cfg.temperature,
        alpha=distill_cfg.alpha,
    )


def build_batch_augmentor(cfg) -> Optional[BatchAugmentor]:
    mix_cfg = cfg.training.mixup
    specmix_cfg = cfg.training.specmix
    if mix_cfg.alpha <= 0 and specmix_cfg.prob <= 0:
        return None
    return BatchAugmentor(
        mixup_alpha=mix_cfg.alpha,
        mixup_prob=mix_cfg.prob,
        specmix_prob=specmix_cfg.prob,
        specmix_segments=specmix_cfg.segments,
        specmix_alpha=specmix_cfg.alpha,
    )


def train_single_fold(
    cfg,
    dataset,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    device: torch.device,
    fold_name: str,
):
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    dataloader_cfg = {}
    if "dataloader" in cfg.training:
        dataloader_cfg = OmegaConf.to_container(cfg.training.dataloader, resolve=True)

    common_loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "collate_fn": collate_batch,
    }
    num_workers = int(dataloader_cfg.get("num_workers", 0))
    pin_memory = bool(dataloader_cfg.get("pin_memory", device.type == "cuda"))
    drop_last = bool(dataloader_cfg.get("drop_last", False))

    train_loader_kwargs = {
        **common_loader_kwargs,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    val_loader_kwargs = {
        **common_loader_kwargs,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        persistent_workers = bool(dataloader_cfg.get("persistent_workers", True))
        prefetch_factor = dataloader_cfg.get("prefetch_factor")
        train_loader_kwargs["persistent_workers"] = persistent_workers
        val_loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            prefetch_factor = int(prefetch_factor)
            train_loader_kwargs["prefetch_factor"] = prefetch_factor
            val_loader_kwargs["prefetch_factor"] = prefetch_factor
    else:
        train_loader_kwargs.setdefault("pin_memory", device.type == "cuda")
        val_loader_kwargs.setdefault("pin_memory", device.type == "cuda")

    sampler_info = build_training_sampler(cfg, dataset, train_indices)
    if sampler_info is not None:
        sampler, sampler_desc = sampler_info
        train_loader_kwargs.pop("shuffle", None)
        train_loader_kwargs["sampler"] = sampler
        print(f"[{fold_name}] Sampler enabled: {sampler_desc}")

    train_loader = DataLoader(train_subset, **train_loader_kwargs)
    val_loader = DataLoader(val_subset, **val_loader_kwargs)
    grad_accum_steps = max(1, int(getattr(cfg.training, "grad_accum_steps", 1)))

    amp_cfg = {}
    if "amp" in cfg.training:
        amp_cfg = OmegaConf.to_container(cfg.training.amp, resolve=True)
    amp_enabled = device.type == "cuda" and bool(amp_cfg.get("enabled", False))
    amp_dtype = torch.bfloat16 if amp_cfg.get("dtype", "float16").lower() == "bfloat16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    autocast_kwargs = {"enabled": amp_enabled}
    if amp_enabled:
        autocast_kwargs["dtype"] = amp_dtype

    class_counts = compute_class_counts(dataset, train_indices, cfg.dataset.num_classes)
    model = build_model(cfg, device)
    model_consumes_features = getattr(model, "consumes_features", True)
    model_requires_waveforms = getattr(model, "requires_waveforms", False)
    if cfg.training.get("compile", False):
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
            except Exception as exc:  # pragma: no cover - optional feature
                print(f"[Warning] torch.compile failed: {exc}")
        else:  # pragma: no cover - torch < 2.0 fallback
            print("[Warning] torch.compile requested but not available in this PyTorch version.")
    criterion = build_loss(cfg, class_counts, device)
    optimizer = create_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)

    augmix_cfg = {}
    if "augmix" in cfg.training:
        augmix_cfg = OmegaConf.to_container(cfg.training.augmix, resolve=True)
    feature_augmentor = FeatureAugmentor(
        freq_mask_param=cfg.training.spec_augment.freq_mask_param,
        time_mask_param=cfg.training.spec_augment.time_mask_param,
        noise_std=cfg.training.feature_aug.noise_std,
        time_shift_pct=cfg.training.feature_aug.time_shift_pct,
        augmix_cfg=augmix_cfg,
    )
    batch_augmentor = build_batch_augmentor(cfg)
    env_extractor = EnvFeatureExtractor(getattr(cfg, "environment", {}), cfg.features.sample_rate)
    open_set_cfg = {}
    if "open_set" in cfg.training:
        open_set_cfg = OmegaConf.to_container(cfg.training.open_set, resolve=True)
    open_set_mixer = OpenSetBatchMixer(
        mix_ratio=open_set_cfg.get("mix_ratio", 0.0),
        noise_std=open_set_cfg.get("noise_std", 0.01),
    ) if open_set_cfg.get("enabled", False) else None
    if open_set_mixer is not None and model_requires_waveforms:
        print(f"[{fold_name}] Open-set mixer disabled for waveform-only model '{cfg.model.name}'.")
        open_set_mixer = None
    open_set_weight = open_set_cfg.get("loss_weight", 0.05)
    adv_cfg = {}
    if "adversarial" in cfg.training:
        adv_cfg = OmegaConf.to_container(cfg.training.adversarial, resolve=True)
    adv_trainer = SpectralPGDAttacker(
        epsilon=adv_cfg.get("epsilon", 0.01),
        steps=adv_cfg.get("steps", 3),
        alpha=adv_cfg.get("alpha", 0.005),
    ) if adv_cfg.get("enabled", False) else None
    if adv_trainer is not None and not model_consumes_features:
        print(f"[{fold_name}] Adversarial trainer disabled (model does not consume feature tensors).")
        adv_trainer = None
    supcon_weight = cfg.training.loss.get("supcon_weight", 0.0)
    supcon_helper = SupConLoss(cfg.training.loss.get("supcon_temperature", 0.07)).to(device) if supcon_weight > 0 else None
    ovr_weight = cfg.training.loss.get("ovr_weight", 0.0)
    ema_model = build_ema(model, cfg.training.ema.decay).to(device) if cfg.training.ema.enabled else None
    distiller = maybe_build_distiller(cfg, device)
    early_stopper = EarlyStopping(
        patience=cfg.training.early_stopping.patience,
        min_delta=cfg.training.early_stopping.min_delta,
        mode="max",
    )
    tta = TTAAugmentor(
        enabled=cfg.evaluation.tta.enabled,
        samples=cfg.evaluation.tta.samples,
        noise_std=cfg.evaluation.tta.noise_std,
        time_shift_pct=cfg.evaluation.tta.time_shift_pct,
    ) if cfg.evaluation.tta.enabled else None

    ckpt_dir = os.path.join(os.getcwd(), "checkpoints", fold_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_model_path = os.path.join(ckpt_dir, "best_model.pth")
    meta_path = os.path.join(ckpt_dir, "best_model_meta.json")
    log_path = os.path.join(ckpt_dir, "training_log.csv")

    log_history: List[Dict[str, float]] = []
    best_metric = float("-inf")
    best_source = "base"
    best_state_cpu: Optional[Dict[str, torch.Tensor]] = None

    latest_state_path = os.path.join(ckpt_dir, "latest_state.pth")

    resume_from_cfg = cfg.training.get("resume_from")
    start_epoch = 1
    if resume_from_cfg:
        resume_root = hydra.utils.to_absolute_path(str(resume_from_cfg))
        resume_candidates = []
        if os.path.isfile(resume_root):
            resume_candidates.append(resume_root)
        if os.path.isdir(resume_root):
            resume_candidates.append(os.path.join(resume_root, "latest_state.pth"))
            resume_candidates.append(os.path.join(resume_root, fold_name, "latest_state.pth"))
        resume_state_path = next((c for c in resume_candidates if c and os.path.isfile(c)), None)
        if resume_state_path:
            print(f"[{fold_name}] Resuming state from {resume_state_path}")
            checkpoint = torch.load(resume_state_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler_state = checkpoint.get("scaler")
            if scaler.is_enabled() and scaler_state is not None:
                scaler.load_state_dict(scaler_state)
            ema_state = checkpoint.get("ema")
            if ema_model is not None and ema_state is not None:
                ema_model.load_state_dict(ema_state)
            early_state = checkpoint.get("early_stopper")
            if early_state:
                early_stopper.best = early_state.get("best", early_stopper.best)
                early_stopper.num_bad_epochs = early_state.get("num_bad_epochs", early_stopper.num_bad_epochs)
            best_metric = checkpoint.get("best_metric", best_metric)
            best_source = checkpoint.get("best_source", best_source)
            log_history = checkpoint.get("log_history", log_history)
            best_state_cpu = checkpoint.get("best_model")
            if best_state_cpu is not None:
                torch.save(best_state_cpu, best_model_path)
            else:
                chk_best_path = checkpoint.get("best_model_path")
                if chk_best_path and os.path.isfile(chk_best_path):
                    best_state_cpu = torch.load(chk_best_path, map_location="cpu")
                    torch.save(best_state_cpu, best_model_path)
                elif os.path.isfile(best_model_path):
                    best_state_cpu = torch.load(best_model_path, map_location="cpu")
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
        else:
            print(f"[{fold_name}] resume_from={resume_from_cfg!r} provided but no checkpoint found.")

    if best_state_cpu is None and os.path.isfile(best_model_path):
        try:
            best_state_cpu = torch.load(best_model_path, map_location="cpu")
        except Exception as exc:
            print(f"[{fold_name}] Warning: failed to preload best model state: {exc}")

    def save_latest(epoch_idx: int) -> None:
        state = {
            "epoch": epoch_idx,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "ema": ema_model.state_dict() if ema_model is not None else None,
            "best_metric": best_metric,
            "best_source": best_source,
            "early_stopper": {
                "best": early_stopper.best,
                "num_bad_epochs": early_stopper.num_bad_epochs,
            },
            "log_history": log_history,
            "best_model": best_state_cpu,
            "best_model_path": best_model_path,
        }
        torch.save(state, latest_state_path)

    num_train_batches = len(train_loader)
    if num_train_batches == 0:
        raise RuntimeError("Training dataloader returned zero batches.")

    print(f"\n--- Training {fold_name} ---")
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        if hasattr(model, "update_freezing"):
            model.update_freezing(epoch)

        model.train()
        total_loss = 0.0
        pbar = train_loader
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            batch_num = batch_idx + 1
            print(f"[{fold_name}] Epoch {epoch:03d} | Batch {batch_num}/{len(pbar)}", end='\r')
            features = batch["features"] if model_consumes_features else None
            labels = batch["label"]
            lengths = batch["length"]
            waveforms = batch.get("waveform")
            waveform_lengths = batch.get("waveform_length")
            env_features = None

            mix_info = None
            if model_consumes_features:
                features = feature_augmentor(features)
                if batch_augmentor is not None:
                    features, labels, mix_info = batch_augmentor(features, labels)
                features = features.to(device, non_blocking=True)
                if mix_info is not None:
                    mix_info = {
                        "y_a": mix_info["y_a"].to(device, non_blocking=True),
                        "y_b": mix_info["y_b"].to(device, non_blocking=True),
                        "lam": mix_info["lam"],
                    }

            labels = labels.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            if waveforms is not None:
                waveforms = waveforms.to(device, non_blocking=True)
            if waveform_lengths is not None:
                waveform_lengths = waveform_lengths.to(device, non_blocking=True)
            if waveforms is not None:
                env_features, _ = env_extractor(waveforms, waveform_lengths, device=device)

            features_for_model = features if model_consumes_features else None
            use_adv = False
            if adv_trainer is not None:
                freq = max(1, adv_cfg.get("frequency", 4))
                use_adv = (batch_idx % freq) == 0
            if use_adv and features_for_model is not None:
                def adv_forward(feat_tensor):
                    return model(
                        features=feat_tensor,
                        waveforms=waveforms,
                        waveform_lengths=waveform_lengths,
                        lengths=lengths,
                        env_features=env_features,
                    )

                features_for_model = adv_trainer(adv_forward, features_for_model, labels)

            with autocast(**autocast_kwargs):
                outputs = model(
                    features=features_for_model,
                    waveforms=waveforms,
                    waveform_lengths=waveform_lengths,
                    lengths=lengths,
                    env_features=env_features,
                )

                if mix_info is not None:
                    loss = mix_info["lam"] * criterion(outputs, mix_info["y_a"]) + (1 - mix_info["lam"]) * criterion(
                        outputs, mix_info["y_b"]
                    )
                else:
                    loss = criterion(outputs, labels)

                if distiller is not None:
                    kd_loss = distiller(
                        outputs,
                        features,
                        waveforms=waveforms,
                        waveform_lengths=waveform_lengths,
                    )
                    loss = (1.0 - distiller.alpha) * loss + distiller.alpha * kd_loss

                if supcon_helper is not None and hasattr(model, "latest_embedding"):
                    sup_loss = supcon_helper(model.latest_embedding, labels)
                    loss = loss + supcon_weight * sup_loss

                if ovr_weight > 0:
                    ovr_targets = F.one_hot(labels, num_classes=cfg.dataset.num_classes).float()
                    ovr_loss = F.binary_cross_entropy_with_logits(outputs, ovr_targets)
                    loss = loss + ovr_weight * ovr_loss

                if open_set_mixer is not None:
                    oe_feats = open_set_mixer(features_for_model.detach())
                    if oe_feats is not None:
                        oe_feats = oe_feats.to(device, non_blocking=True)
                        oe_logits = model(
                            features=oe_feats,
                            waveforms=None,
                            waveform_lengths=None,
                            lengths=None,
                            env_features=None,
                        )
                        margin = open_set_cfg.get("margin", 5.0)
                        oe_energy = torch.logsumexp(oe_logits, dim=-1)
                        loss = loss + open_set_weight * torch.relu(oe_energy - margin).mean()

            total_loss += loss.detach().item()

            loss_for_backward = loss / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            step_now = (batch_num % grad_accum_steps == 0) or (batch_num == num_train_batches)
            if step_now:
                if scaler.is_enabled():
                    if cfg.training.gradient_clip:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if cfg.training.gradient_clip:
                        clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
                    optimizer.step()

                if ema_model is not None:
                    ema_model.update_parameters(model)

                optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        train_loss = total_loss / max(1, len(train_loader))

        base_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            env_extractor=env_extractor,
        )
        candidate_metrics = base_metrics
        candidate_state = model.state_dict()
        candidate_source = "base"

        if ema_model is not None:
            refresh_batch_norm(ema_model, train_loader, device)
            ema_metrics = evaluate(
                ema_model,
                val_loader,
                criterion,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                env_extractor=env_extractor,
            )
            if ema_metrics["uar"] >= base_metrics["uar"]:
                candidate_metrics = ema_metrics
                candidate_state = ema_model.state_dict()
                candidate_source = "ema"

        log_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": candidate_metrics["loss"],
                "val_uar": candidate_metrics["uar"],
                "val_macro_f1": candidate_metrics["macro_f1"],
                "source": candidate_source,
            }
        )

        print(
            f"[{fold_name}] Epoch {epoch:03d} | Train {train_loss:.4f} | "
            f"Val UAR {candidate_metrics['uar']:.4f} | Val F1 {candidate_metrics['macro_f1']:.4f} | Source {candidate_source}"
        )

        if candidate_metrics["uar"] > best_metric:
            best_metric = candidate_metrics["uar"]
            best_source = candidate_source
            state_to_save = {k: v.detach().cpu() for k, v in candidate_state.items()}
            torch.save(state_to_save, best_model_path)
            best_state_cpu = state_to_save
            print(f"  -> New best ({best_source.upper()}) model saved to {best_model_path}")

        save_latest(epoch)

        if early_stopper.step(candidate_metrics["uar"]):
            print(f"[{fold_name}] Early stopping triggered.")
            break

    if start_epoch > cfg.training.epochs:
        save_latest(start_epoch - 1)

    pd.DataFrame(log_history).to_csv(log_path, index=False)

    if not os.path.isfile(best_model_path):
        if best_state_cpu is not None:
            torch.save(best_state_cpu, best_model_path)
        else:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, best_model_path)

    best_model = build_model(cfg, device)
    best_state = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(best_state)
    if cfg.training.get("compile", False) and hasattr(torch, "compile"):
        try:
            best_model = torch.compile(best_model)
        except Exception as exc:  # pragma: no cover - optional feature
            print(f"[Warning] torch.compile (eval) failed: {exc}")

    final_metrics = evaluate(
        best_model,
        val_loader,
        criterion,
        device,
        tta=None,
        collect_logits=cfg.evaluation.calibration.enabled,
        collect_embeddings=cfg.evaluation.calibration.enabled,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        env_extractor=env_extractor,
    )

    calibration_payload = {}
    density_payload = {}
    conformal_payload = {}
    threshold_payload = {
        "profile": env_extractor.summarize_profile(),
        "params": OmegaConf.to_container(getattr(cfg.environment, "thresholds", {}), resolve=True)
        if hasattr(cfg, "environment") and "thresholds" in cfg.environment
        else {},
    }

    if cfg.evaluation.calibration.enabled:
        logits = final_metrics.pop("logits").to(device)
        labels_tensor = final_metrics.pop("labels").to(device)
        embeddings_tensor = final_metrics.pop("embeddings").to(device)
        temperature = tune_temperature(logits, labels_tensor)
        temperature_value = temperature.item()
        print(f"[{fold_name}] Temperature calibrated: {temperature_value:.4f}")

        dirichlet = DirichletCalibrator(cfg.dataset.num_classes)
        dirichlet.fit(logits, labels_tensor)
        calibrated_probs = dirichlet(logits)
        iso = IsotonicBackoff()
        top_conf, preds = calibrated_probs.max(dim=-1)
        iso.fit(top_conf, preds.eq(labels_tensor))
        ece_score = compute_ece(calibrated_probs, labels_tensor)
        calib_bundle = CalibrationArtifactBundle(
            dirichlet_state=dirichlet.state_dict(),
            isotonic_state=iso.state_dict(),
            metrics={"ece": ece_score},
        )
        calibration_payload = {
            "dirichlet": {k: v.cpu().tolist() if torch.is_tensor(v) else v for k, v in calib_bundle.dirichlet_state.items()},
            "isotonic": {k: v.cpu().tolist() if torch.is_tensor(v) else v for k, v in calib_bundle.isotonic_state.items()},
            "metrics": calib_bundle.metrics,
        }

        ood_cfg = getattr(cfg, "ood", {})
        gscore = GScoreEstimator(
            alpha=ood_cfg.get("alpha", 1.0),
            beta=ood_cfg.get("beta", 0.5),
            shrinkage=ood_cfg.get("shrinkage", 0.1),
        )
        gscore.fit(embeddings_tensor, labels_tensor)
        scores = gscore.gscore(logits, embeddings_tensor)
        tau = gscore.conformal_threshold(scores, ood_cfg.get("conformal_q", 0.02))
        density_payload = gscore.to_dict()
        conformal_payload = {
            "quantile": ood_cfg.get("conformal_q", 0.02),
            "threshold": tau,
            "temperature": gscore.energy_gate.temperature,
        }
    else:
        temperature = torch.tensor(1.0, device=device)
        temperature_value = 1.0

    tta_metrics = evaluate(
        best_model,
        val_loader,
        criterion,
        device,
        tta=tta,
        temperature=temperature,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        env_extractor=env_extractor,
    )

    signature_dir = os.path.join(ckpt_dir, "signatures")
    signature_package = SignaturePackage(
        threshold_profile=threshold_payload,
        conformal=conformal_payload,
        density_stats=density_payload,
        calibration=calibration_payload,
    )
    SignatureWriter(signature_dir).write(signature_package)

    meta = {
        "fold": fold_name,
        "best_source": best_source,
        "temperature": temperature_value,
        "val_loss": tta_metrics["loss"],
        "val_uar": tta_metrics["uar"],
        "val_macro_f1": tta_metrics["macro_f1"],
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[{fold_name}] Final | Loss {tta_metrics['loss']:.4f} | UAR {tta_metrics['uar']:.4f} | "
        f"Macro F1 {tta_metrics['macro_f1']:.4f}"
    )

    return {
        "metrics": tta_metrics,
        "meta_path": meta_path,
        "checkpoint": best_model_path,
        "log_path": log_path,
    }


def build_loco_splits(cfg, dataset) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    loco_cfg = cfg.dataset.get("loco")
    if not loco_cfg or not loco_cfg.get("enabled", False):
        return None
    dataset_names = np.array([sample.get("dataset", "unknown") for sample in dataset.samples])
    unique_names = sorted(set(dataset_names.tolist()))
    holdouts = loco_cfg.get("holdouts") or unique_names
    splits = []
    for holdout in holdouts:
        mask = dataset_names == holdout
        if not mask.any():
            continue
        val_idx = np.where(mask)[0]
        train_idx = np.where(~mask)[0]
        if len(val_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    if not splits:
        return None
    return splits


def build_cv_splits(cfg, dataset) -> List[Tuple[np.ndarray, np.ndarray]]:
    labels = np.array([sample["label"] for sample in dataset.samples])
    speakers = np.array([sample["speaker_id"] for sample in dataset.samples])

    if cfg.evaluation.cv.enabled:
        n_splits = cfg.evaluation.cv.folds
        if StratifiedGroupKFold is not None and cfg.evaluation.cv.stratified:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=cfg.evaluation.cv.shuffle,
                random_state=cfg.seed,
            )
            splits = splitter.split(labels, labels, groups=speakers)
        else:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=cfg.evaluation.cv.shuffle,
                random_state=cfg.seed,
            )
            splits = splitter.split(labels, labels)
        return [(np.array(train_idx), np.array(val_idx)) for train_idx, val_idx in splits]

    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.evaluation.test_size, random_state=cfg.seed)
    train_indices, val_indices = next(gss.split(dataset.samples, groups=speakers))
    return [(np.array(train_indices), np.array(val_indices))]


def run_training(cfg: DictConfig) -> List[Dict[str, object]]:
    print("--- Configuration ---\n" + OmegaConf.to_yaml(cfg) + "---------------------")
    deterministic = cfg.training.get("deterministic", True) if "training" in cfg else True
    set_seed(cfg.seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        allow_tf32 = cfg.training.get("allow_tf32", True)
        try:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
        except AttributeError:  # pragma: no cover - older torch fallback
            pass
        matmul_precision = cfg.training.get("matmul_precision", None)
        if matmul_precision:
            try:
                torch.set_float32_matmul_precision(matmul_precision)
            except AttributeError:  # pragma: no cover - torch < 1.12
                print(
                    f"[Warning] torch.set_float32_matmul_precision unavailable; "
                    f"requested precision '{matmul_precision}' will be ignored."
                )
    print(f"Using device: {device}")

    dataset = build_dataset(cfg)

    splits = build_loco_splits(cfg, dataset) or build_cv_splits(cfg, dataset)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_name = f"fold_{fold_idx}" if cfg.evaluation.cv.enabled else "holdout"
        result = train_single_fold(cfg, dataset, train_idx, val_idx, device, fold_name)
        fold_results.append(result)

    if len(fold_results) > 1:
        uars = [res["metrics"]["uar"] for res in fold_results]
        f1s = [res["metrics"]["macro_f1"] for res in fold_results]
        print("\n=== Cross-Validation Summary ===")
        print(f"Mean UAR : {np.mean(uars):.4f} ± {np.std(uars):.4f}")
        print(f"Mean F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    print("\n--- Training Complete ---")
    for res in fold_results:
        print(f"Checkpoint: {res['checkpoint']}")
        print(f"Metadata  : {res['meta_path']}")
        print(f"Log       : {res['log_path']}")

    return fold_results


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
