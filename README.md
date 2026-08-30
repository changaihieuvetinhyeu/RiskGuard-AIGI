# RiskGuard-AIGI

Pipeline for selective AI-generated image detection, error-risk estimation, acceptance-policy certification, analysis, and release auditing.

## Requirements

- Python 3.10 or newer
- A CUDA-capable GPU for full detector inference
- The official detector repositories and checkpoints expected by the adapters
- Prepared manifests, cached detector outputs, experiment configuration, and evaluation inputs at the relative locations expected by the scripts

Create an environment and install the runtime packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn pyyaml pillow pyarrow matplotlib seaborn psutil pytest torch torchvision faiss-cpu
export PYTHONPATH="$PWD/src"
```

## Prepare source data

```bash
PYTHONPATH=src python scripts/summarize_archive_inventory.py logs/genimage_drive_inventory.json
PYTHONPATH=src python scripts/inspect_remote_archive_sizes.py
PYTHONPATH=src python scripts/resolve_duplicate_archive_paths.py
PYTHONPATH=src python scripts/estimate_archive_storage.py
PYTHONPATH=src python scripts/plan_multi_disk_archive_layout.py
bash scripts/download_archives_to_multiple_disks.sh
PYTHONPATH=src python scripts/verify_archive_downloads.py
PYTHONPATH=src python scripts/inspect_image_integrity.py --root datasets/processed/genimage
PYTHONPATH=src python scripts/check_detector_dependencies.py
```

## Dataset download

The experiments use [GenImage](https://github.com/GenImage-Dataset/GenImage) as the main image corpus and the [B-Free viral-image dataset](https://github.com/grip-unina/B-Free/tree/main/viral_images_dataset) as an external evaluation set. Review each source's license and terms before downloading or redistributing its files.

### GenImage

The official GenImage archives are hosted in this [Google Drive folder](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS?usp=sharing). The archive set is approximately 660 GB before extraction, so allow substantially more free space for the extracted images.

For a browser download, open the folder link and download every generator directory. For a resumable command-line download, install `rclone`, configure a Google Drive remote named `gdrive`, and run:

```bash
mkdir -p datasets/raw/genimage
rclone config

export GENIMAGE_FOLDER_ID="1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS"
rclone copy gdrive: datasets/raw/genimage \
  --drive-root-folder-id "$GENIMAGE_FOLDER_ID" \
  --progress \
  --transfers 4 \
  --checkers 8

rclone check gdrive: datasets/raw/genimage \
  --drive-root-folder-id "$GENIMAGE_FOLDER_ID" \
  --one-way
```

Keep each `.zip` file beside all of its `.z01`, `.z02`, and later split parts. On Ubuntu, install `7zip` support and extract every archive into a generator-specific directory:

```bash
sudo apt-get update
sudo apt-get install -y p7zip-full

mkdir -p datasets/processed/genimage
find datasets/raw/genimage -type f -name '*.zip' -print0 |
while IFS= read -r -d '' archive; do
  generator="$(basename "$(dirname "$archive")")"
  output="datasets/processed/genimage/$generator"
  mkdir -p "$output"
  7z x "$archive" "-o$output"
done
```

Test an archive before extraction with `7z t path/to/archive.zip`. If a split part is absent or damaged, download that part again before continuing.



```
The prepared manifests contain a `physical_output_path` column. Ensure every value points to the corresponding extracted image on the current machine. Place the manifests, prediction caches, experiment configuration, and required artifacts in the relative locations expected by the scripts. Each executable also supports `--help` where command-line options are available.

## Detector code and checkpoints

The adapters expect the official repositories at these locations:

```bash
mkdir -p third_party
git clone https://github.com/Ouxiang-Li/SAFE.git third_party/SAFE
git clone https://github.com/WisconsinAIVision/UniversalFakeDetect.git third_party/UniversalFakeDetect
```

Confirm that these checkpoint files exist before inference:

```text
third_party/SAFE/checkpoint/checkpoint-best.pth
third_party/UniversalFakeDetect/pretrained_weights/fc_weights.pth
```

## Verify the installation

```bash
PYTHONPATH=src python -m pytest -q
```

## Fit and evaluate selective baselines

```bash
PYTHONPATH=src python scripts/fit_selective_baselines.py
PYTHONPATH=src python scripts/score_selective_baselines.py
PYTHONPATH=src python scripts/select_baseline_thresholds.py
PYTHONPATH=src python scripts/evaluate_selective_baselines.py
PYTHONPATH=src python scripts/audit_selective_baselines.py
PYTHONPATH=src python scripts/audit_selective_baselines_extended.py
```

## Build transformation reliability features

```bash
PYTHONPATH=src python scripts/build_transformation_features.py prepare --scope full --audit-parents 256 --manifest-workers 8
PYTHONPATH=src python scripts/build_transformation_features.py infer --scope full --detector safe --device cuda:0 --batch-size 192 --image-workers 8 --shard-size 5000
PYTHONPATH=src python scripts/build_transformation_features.py infer --scope full --detector univfd --device cuda:0 --batch-size 768 --image-workers 8 --shard-size 5000
PYTHONPATH=src python scripts/build_transformation_features.py features --scope full --detector safe --device cuda:0 --support-batch-size 8192
PYTHONPATH=src python scripts/build_transformation_features.py features --scope full --detector univfd --device cuda:0 --support-batch-size 8192
PYTHONPATH=src python scripts/build_transformation_features.py determinism --scope full --parents 2048
PYTHONPATH=src python scripts/build_transformation_features.py audit --scope full
PYTHONPATH=src python scripts/audit_transformation_features.py
```

Adjust batch sizes and worker counts to match the available GPU memory, CPU capacity, and storage throughput.

## Fit and evaluate the error calibrator

```bash
PYTHONPATH=src python scripts/verify_calibrator_inputs.py
PYTHONPATH=src python scripts/build_calibrator_cross_validation_folds.py --force
PYTHONPATH=src python scripts/audit_calibrator_feature_values.py
PYTHONPATH=src python scripts/fit_error_calibrator.py --force
PYTHONPATH=src python scripts/run_feature_ablations.py --force
PYTHONPATH=src python scripts/score_error_risk.py --force
PYTHONPATH=src python scripts/evaluate_error_calibrator.py --force
PYTHONPATH=src python scripts/audit_error_calibrator.py --force
PYTHONPATH=src python scripts/audit_error_calibrator_extended.py --force
```

## Select and certify acceptance policies

```bash
PYTHONPATH=src python scripts/build_policy_calibration_split.py
PYTHONPATH=src python scripts/select_acceptance_candidates.py
PYTHONPATH=src python scripts/freeze_acceptance_policies.py
PYTHONPATH=src python scripts/certify_acceptance_policies.py
PYTHONPATH=src python scripts/evaluate_certified_policies.py
PYTHONPATH=src python scripts/bootstrap_certification_results.py
PYTHONPATH=src python scripts/audit_risk_certification.py
```

## Run analyses and release checks

```bash
PYTHONPATH=src python scripts/compare_feature_sets.py
PYTHONPATH=src python scripts/run_support_and_drift_ablation.py
PYTHONPATH=src python scripts/run_logit_trajectory_ablation.py
PYTHONPATH=src python scripts/promote_selected_feature_model.py
PYTHONPATH=src python scripts/audit_error_analysis.py --stage run_all
PYTHONPATH=src python scripts/audit_fail_closed_behavior.py
PYTHONPATH=src python scripts/certify_margin_only_baseline.py
PYTHONPATH=src python scripts/plot_risk_coverage.py
PYTHONPATH=src python scripts/audit_release_readiness.py --stage run_all
```

Use `--resume` for supported long-running jobs, and use `--force` only when intentionally rebuilding existing outputs.
