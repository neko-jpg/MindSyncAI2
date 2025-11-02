import os
import torch
import hydra
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import recall_score, f1_score
from torch.nn import CrossEntropyLoss

from data.ravdess_dataset import RavdessDataset
from .models.mobile_crnn_v1 import MobileCRNNv1

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_batch(batch):
    """Pad variable-length spectrograms and stack other fields."""
    features = [item["features"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    speaker_ids = torch.tensor([item["speaker_id"] for item in batch], dtype=torch.long)

    lengths = torch.tensor([feat.shape[-1] for feat in features], dtype=torch.long)
    max_length = int(lengths.max())

    padded = []
    for feat in features:
        pad_amount = max_length - feat.shape[-1]
        if pad_amount > 0:
            feat = F.pad(feat, (0, pad_amount))
        padded.append(feat)

    feature_tensor = torch.stack(padded)
    return {
        "features": feature_tensor,
        "label": labels,
        "speaker_id": speaker_ids,
        "length": lengths
    }


def train_one_epoch(model, ema_model, data_loader, criterion, optimizer, device, spec_augment=None):
    """Runs a single training epoch and updates the EMA model."""
    model.train()
    total_loss = 0
    pbar = tqdm(data_loader, desc="Training", leave=False)
    for batch in pbar:
        features = batch['features']
        if spec_augment is not None:
            features = spec_augment(features)
        features = features.to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Update EMA model after each step
        ema_model.update_parameters(model)

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)

@torch.no_grad()
def evaluate(model_to_eval, data_loader, criterion, device):
    """Evaluates a given model (can be the base model or EMA model)."""
    model_to_eval.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(data_loader, desc="Evaluating", leave=False)
    for batch in pbar:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        outputs = model_to_eval(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, uar, macro_f1

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("--- Configuration ---\n" + OmegaConf.to_yaml(cfg) + "---------------------")
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = hydra.utils.to_absolute_path(cfg.data_dir)
    sample_rate = OmegaConf.select(cfg, "features.sample_rate", default=16000)
    n_mels = OmegaConf.select(cfg, "features.n_mels", default=64)

    full_dataset = RavdessDataset(data_dir=data_path, sample_rate=sample_rate, n_mels=n_mels)

    speaker_ids = np.array([s['speaker_id'] for s in full_dataset.samples])
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.evaluation.test_size, random_state=cfg.seed)
    train_indices, val_indices = next(gss.split(full_dataset.samples, groups=speaker_ids))
    train_dataset, val_dataset = Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    model = MobileCRNNv1(num_classes=cfg.dataset.num_classes).to(device)
    # Initialize EMA model
    ema_model = AveragedModel(model)
    ema_model.module.to(device)

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    spec_augment = None
    if OmegaConf.select(cfg, "training.spec_augment.freq_mask_param", default=None) is not None:
        spec_augment = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=cfg.training.spec_augment.freq_mask_param),
            T.TimeMasking(time_mask_param=cfg.training.spec_augment.time_mask_param),
        )

    log_history = []
    best_uar = 0.0

    print("\n--- Starting Training ---")
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_one_epoch(model, ema_model, train_loader, criterion, optimizer, device, spec_augment)

        # Evaluate the base model for most of the training
        val_loss, uar, macro_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_log = { "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "uar": uar, "macro_f1": macro_f1 }
        log_history.append(epoch_log)

        print(
            f"Epoch {epoch:02d}/{cfg.training.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"UAR: {uar:.4f} | Macro F1: {macro_f1:.4f}"
        )

        # Save the best *base* model during training
        if uar > best_uar:
            best_uar = uar
            ema_model_path_run = os.path.join(os.getcwd(), "best_model.pth")
            ema_model_path_project = hydra.utils.to_absolute_path("best_model.pth")
            torch.save(ema_model.module.state_dict(), ema_model_path_run)
            torch.save(ema_model.module.state_dict(), ema_model_path_project)
            print(f"  -> New best EMA model saved with UAR: {uar:.4f}")

    # --- Final Evaluation of the EMA model ---
    print("\n--- Final Evaluation of EMA Model ---")
    # Manually update bn statistics for the EMA model
    print("Updating BatchNorm statistics for the EMA model...")
    ema_model.train() # Set to train mode to update BN stats
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Updating BN", leave=False):
            features = batch['features'].to(device)
            # Forward pass to update running mean/var of BN layers
            ema_model(features)

    val_loss_ema, uar_ema, macro_f1_ema = evaluate(ema_model, val_loader, criterion, device)
    print(
        f"Final EMA Model Performance | Val Loss: {val_loss_ema:.4f} | "
        f"UAR: {uar_ema:.4f} | Macro F1: {macro_f1_ema:.4f}"
    )
    # Also log this final performance
    final_log = {"epoch": "final_ema", "train_loss": -1, "val_loss": val_loss_ema, "uar": uar_ema, "macro_f1": macro_f1_ema}
    log_history.append(final_log)

    pd.DataFrame(log_history).to_csv("training_log.csv", index=False)
    print(f"\n--- Training Complete ---\nOutput directory: {os.getcwd()}")

if __name__ == "__main__":
    main()
