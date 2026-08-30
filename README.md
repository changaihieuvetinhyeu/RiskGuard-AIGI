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

The repository contains code only. Restore the companion inputs, checkpoints, and configuration before running the full workflow. Each executable also supports `--help` where command-line options are available.

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
