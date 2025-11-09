# Speech Emotion Recognition with PyTorch

This repository provides a complete pipeline for training and exporting a Speech Emotion Recognition (SER) model. The entire workflow, from feature extraction to model training and deployment, is implemented in PyTorch. The project is designed to be extensible, allowing for the easy integration of new datasets and model architectures.

## Key Features

- **End-to-End PyTorch Workflow**: The entire pipeline, including feature extraction, training, and inference, is built with PyTorch and `torchaudio`.
- **Unified Preprocessing**: A consistent and unified feature extraction process (`log-mel spectrogram`) is used for both training and inference, eliminating inconsistencies.
- **Extensible Data Loaders**: The data loading structure is designed to be modular, allowing for the easy addition of new datasets like RAVDESS and CREMA-D.
- **Mobile-First Model**: Includes a lightweight, mobile-first CRNN model (`MobileCRNNv1`) designed for efficient on-device inference.
- **PyTorch Mobile Export**: Provides a script to export the trained model to the PyTorch Mobile (TorchScript) format for deployment on iOS and Android.

## Project Structure

```
.
├── export/                  # Scripts for exporting the model
│   └── export_torchscript.py
├── ser/                     # Main source code for the SER project
│   ├── conf/                # Hydra configuration files
│   ├── data/                # Dataset loaders
│   ├── models/              # Model architectures
│   ├── __init__.py
│   ├── main.py              # Main training script
│   └── preproc.py           # Unified preprocessing module
├── test/                    # Tests for the project
├── models/                  # Saved models
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 1. Setup

### a. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### b. Install Dependencies
It is recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.
```bash
pip install -r requirements.txt
```

### c. Prepare Datasets
Bootstrap the multi-corpus pack (RAVDESS, CREMA-D, TESS, SAVEE, Emo-DB) from Hugging Face mirrors:
```bash
python scripts/dataset_setup.py --root data
```

This yields:
```text
.
data/
|- RAVDESS/
|- CREMA-D/
|- TESS/
|- SAVEE/
- EMO-DB/
```

Prefer manual downloads? Place the corpora under the same layout and the loaders will discover them automatically.


## 2. Training the Model

The training process is managed by Hydra. Use either the baseline configuration (ser/conf/config.yaml) or the Phase 2 upgrade (ser/conf/config_phase2.yaml).

### a. Baseline quick start
```bash
python -m ser.main
```
Override parameters with standard Hydra overrides:
```bash
python -m ser.main training.epochs=50 dataset.name=ravdess
```

### b. Phase 2 orchestrator
```bash
python phase2_train.py --config-name config_phase2 --keepalive-seconds 300 --log-file outputs/phase2_train.log
```
Add -o KEY=VALUE for Hydra overrides (for example -o training.epochs=160).

**RTX 3080 8GB preset**  
Use `--config-name config_phase2_rtx3080` to load the memory-optimized preset (micro-batch 8 with 4-step gradient accumulation). Example:
```bash
python phase2_train.py --config-name config_phase2_rtx3080 -o model.use_mixstyle=true -o model.use_conformer=true
```

If a run stops mid-way, resume from the latest checkpoints with:
```bash
python phase2_train.py --config-name config_phase2 --resume-from checkpoints
```
The --resume-from flag accepts either a checkpoint file (latest_state.pth) or a directory containing per-fold subdirectories.

Every run stores the best-scoring model (by UAR) as best_model.pth and writes epoch metrics to training_log.csv.

### c. Self-training with pseudo labels
1. Generate confident predictions on unlabeled audio:
   ```bash
   python scripts/generate_pseudo_labels.py \
     --config-name config_phase2 \
     --checkpoint checkpoints/fold_0/best_model.pth \
     --unlabeled-root /path/to/unlabeled_audio \
     --output-csv data/pseudo/pseudo_labels.csv \
     --copy-audio data/pseudo/audio \
     --min-confidence 0.9
   ```
   The script mirrors accepted clips (optional) and emits `pseudo_labels.csv` plus a `.meta` summary.
2. Append the pseudo dataset when launching training:
   ```bash
   python phase2_train.py \
     --config-name config_phase2 \
     -o dataset.names='[ravdess,cremad,tess,savee,emodb,pseudo]' \
     -o dataset.sources.pseudo.type=metadata \
     -o dataset.sources.pseudo.metadata_path=data/pseudo/pseudo_labels.csv \
     -o dataset.sources.pseudo.audio_root=data/pseudo/audio
   ```
   Additional overrides (e.g. lower `min_confidence`, adjusted loss weights) can be stacked as needed.
   Tip: enable balanced multi-corpus sampling via `-o dataset.sampling.strategy=balanced` or set per-source weights with `-o dataset.sources.NAME.weight=1.5`.

## 3. Exporting the Model for Mobile

To deploy the model on a mobile device, you need to convert it to the PyTorch Mobile (TorchScript) format.

Run the `export_torchscript.py` script, pointing it to the checkpoint of your trained model:
```bash
python export/export_torchscript.py --checkpoint_path best_model.pth --output_path models/mobile_crnn_v1.ptl
```

The final, optimized model will be saved at `models/mobile_crnn_v1.ptl`, ready to be integrated into an iOS or Android application using the PyTorch Mobile SDK.
