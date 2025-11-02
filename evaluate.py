import torch
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import recall_score, f1_score, accuracy_score
import torch.nn.functional as F

# Add project root to path to allow importing 'ser' modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.ravdess_dataset import RavdessDataset
from ser.models.mobile_crnn_v1 import MobileCRNNv1


def collate_batch(batch):
    """Pad variable-length spectrograms in the batch to the same temporal length."""
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

@torch.no_grad()
def evaluate_model(model, data_loader, device):
    """Evaluates the model and returns key metrics."""
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(data_loader, desc="Evaluating"):
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        outputs = model(features)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        "Accuracy": accuracy,
        "UAR": uar,
        "Macro_F1": macro_f1
    }

@hydra.main(config_path="ser/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("--- PyTorch Model Evaluation ---")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    data_path = hydra.utils.to_absolute_path(cfg.data_dir)
    full_dataset = RavdessDataset(data_dir=data_path, sample_rate=16000, n_mels=64)

    speaker_ids = np.array([s['speaker_id'] for s in full_dataset.samples])
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.evaluation.test_size, random_state=cfg.seed)
    _, val_indices = next(gss.split(full_dataset.samples, groups=speaker_ids))
    val_dataset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_batch
    )

    print(f"Validation set size: {len(val_dataset)}")

    # --- Model Loading ---
    model = MobileCRNNv1(num_classes=cfg.dataset.num_classes).to(device)
    model_path = hydra.utils.to_absolute_path("best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully from '{model_path}'")

    # --- Evaluation ---
    metrics = evaluate_model(model, val_loader, device)

    print("\n--- Evaluation Results ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("--------------------------")

if __name__ == "__main__":
    main()
