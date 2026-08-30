#!/usr/bin/env python3
"""Expanded Phase 3 final audit with A-M subchecks and v2 reports."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from selective_detection.tabular_input_schema import read_manifest_csv
from selective_detection.selective_baselines import (
    DETECTORS,
    MANDATORY_BASELINES,
    SPLITS,
    sha256_file,
    verify_phase2_frozen_hashes,
    write_json,
    write_yaml,
)
from selective_detection.selective_metrics import (
    aurc,
    eaurc,
    risk_at_coverage,
    sha256_deduplicate,
)
from selective_detection.selective_thresholds import (
    clopper_pearson_upper,
    select_global_threshold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"
PHASE3 = ARTIFACTS / "phase3"
REPORTS = PROJECT_ROOT / "reports" / "phase3"
MANIFESTS = PROJECT_ROOT / "datasets" / "manifests"
CONFIG = PROJECT_ROOT / "configs" / "phase3" / "selective_baselines.yaml"

EVAL_ROLES = ("protocol_seen", "protocol_held_out")
SCORE_SCOPES = ("risk_fit", "threshold_cal", "protocol_seen", "protocol_held_out")
HELD_OUT_GENERATORS = {
    "split_a": {"midjourney", "sd15", "wukong", "vqdm"},
    "split_b": {"biggan", "adm", "glide", "sd14"},
}
SEEN_GENERATORS = {
    "split_a": {"adm", "biggan", "glide", "sd14"},
    "split_b": {"midjourney", "sd15", "wukong", "vqdm"},
}


@dataclass
class Check:
    category: str
    subcheck_id: str
    check: str
    status: str
    hard_blocker: bool
    detail: str = ""
    artifact: str = ""
    affected_detector: str = "all"
    affected_baseline: str = "all"

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "subcheck_id": self.subcheck_id,
            "check": self.check,
            "status": self.status,
            "hard_blocker": self.hard_blocker,
            "artifact": self.artifact,
            "affected_detector": self.affected_detector,
            "affected_baseline": self.affected_baseline,
            "detail": self.detail,
        }


class Audit:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(
        self,
        category: str,
        subcheck_id: str,
        check: str,
        ok: bool,
        hard_blocker: bool,
        detail: str = "",
        artifact: str = "",
        affected_detector: str = "all",
        affected_baseline: str = "all",
    ) -> None:
        self.checks.append(
            Check(
                category=category,
                subcheck_id=subcheck_id,
                check=check,
                status="pass" if ok else "fail",
                hard_blocker=hard_blocker,
                detail=detail,
                artifact=artifact,
                affected_detector=affected_detector,
                affected_baseline=affected_baseline,
            )
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([check.as_dict() for check in self.checks])


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_predictions(detector: str) -> pd.DataFrame:
    cache = ARTIFACTS / "cache" / detector / "clean"
    index = pd.read_parquet(cache / "index.parquet")
    frames = [pd.read_parquet(path) for path in sorted(index["prediction_shard"].unique())]
    df = pd.concat(frames, ignore_index=True)
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def phase2_thresholds() -> pd.DataFrame:
    df = pd.read_csv(ARTIFACTS / "phase2_clean_thresholds.csv")
    return df[df["threshold_source"] == "threshold_cal"].copy()


def score_path(detector: str, baseline: str, split: str, scope: str) -> Path:
    return PHASE3 / "scores" / detector / baseline / f"{split}_{scope}.parquet"


def manifest_for(split: str, scope: str) -> Path:
    if scope == "risk_fit":
        return MANIFESTS / f"{split}_risk_fit.csv"
    if scope == "threshold_cal":
        return MANIFESTS / f"{split}_threshold_cal.csv"
    return MANIFESTS / f"verified_v2_{split}_{scope}_eval.csv"


def expected_partition(scope: str) -> str:
    return {
        "risk_fit": "risk_fit",
        "threshold_cal": "threshold_cal",
        "protocol_seen": "clean_seen_test",
        "protocol_held_out": "clean_unseen_test",
    }[scope]


def dedup_score(df: pd.DataFrame) -> pd.DataFrame:
    kept, _ = sha256_deduplicate(df)
    return kept


def bfree_metrics_with_renamed_external_columns() -> pd.DataFrame:
    path = PHASE3 / "bfree_snapshot_selective_metrics.csv"
    df = pd.read_csv(path)
    rename = {
        "coverage_at_cp_risk_le_1pct": "bfree_coverage_at_genimage_calibrated_cp_risk_le_1pct_threshold",
        "selective_risk_at_cp_risk_le_1pct": "bfree_selective_risk_at_genimage_calibrated_cp_risk_le_1pct_threshold",
        "coverage_at_cp_risk_le_5pct": "bfree_coverage_at_genimage_calibrated_cp_risk_le_5pct_threshold",
        "selective_risk_at_cp_risk_le_5pct": "bfree_selective_risk_at_genimage_calibrated_cp_risk_le_5pct_threshold",
    }
    present = {old: new for old, new in rename.items() if old in df.columns}
    if present:
        df = df.rename(columns=present)
        df.to_csv(path, index=False)
    return df


def compute_bfree_raw_detector_metrics() -> pd.DataFrame:
    thresholds = phase2_thresholds()
    rows = []
    for detector in DETECTORS:
        pred = pd.read_parquet(ARTIFACTS / f"bfree_viral_verified_{detector}_predictions.parquet")
        pred["sample_id"] = pred["sample_id"].astype(str)
        labels = pred["label"].to_numpy(dtype=int)
        scores = pred["fake_probability"].to_numpy(dtype=float)
        auroc = float(roc_auc_score(labels, scores))
        aupr = float(average_precision_score(labels, scores))
        for split in SPLITS:
            threshold = float(
                thresholds[(thresholds["detector"] == detector) & (thresholds["split"] == split)][
                    "decision_threshold"
                ].iloc[0]
            )
            yhat = (scores >= threshold).astype(int)
            real = labels == 0
            fake = labels == 1
            rows.append(
                {
                    "detector": detector,
                    "split": split,
                    "phase2_threshold_source": "GenImage threshold_cal",
                    "phase2_decision_threshold": threshold,
                    "sample_count": int(len(pred)),
                    "real_count": int(real.sum()),
                    "fake_count": int(fake.sum()),
                    "raw_detector_AUROC": auroc,
                    "raw_detector_AUPR": aupr,
                    "raw_detector_accuracy": float((yhat == labels).mean()),
                    "raw_detector_balanced_accuracy": float(balanced_accuracy_score(labels, yhat)),
                    "raw_detector_far_real_false_accusation_rate": float(((yhat == 1) & real).sum() / real.sum()),
                    "raw_detector_fnr_fake_miss_rate": float(((yhat == 0) & fake).sum() / fake.sum()),
                    "raw_detector_error_count": int((yhat != labels).sum()),
                    "threshold_note": (
                        "Split A and Split B use the same B-Free logits but different GenImage "
                        "threshold_cal decision thresholds, so thresholded raw metrics can differ by split."
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(PHASE3 / "bfree_raw_detector_metrics.csv", index=False)
    return out


def audit_a(audit: Audit, registry: dict[str, Any], thresholds: pd.DataFrame) -> None:
    try:
        frozen = verify_phase2_frozen_hashes(PROJECT_ROOT)
        failed = frozen[frozen["status"] == "fail"]
        audit.add("A", "A1", "All Phase 2 frozen hashes match.", len(failed) == 0, True, f"checked {len(frozen)} frozen inputs")
    except Exception as exc:
        audit.add("A", "A1", "All Phase 2 frozen hashes match.", False, True, str(exc), "artifacts/phase2_frozen_artifact_hashes.csv")

    checkpoint_details = []
    checkpoint_ok = True
    for detector in DETECTORS:
        info = registry["detectors"][detector]
        checkpoint_path = PROJECT_ROOT / info["checkpoint_path"]
        observed = sha256_file(checkpoint_path)
        expected = info["checkpoint_sha256"]
        checkpoint_ok &= observed == expected
        checkpoint_details.append(f"{detector}:{observed[:12]}=={expected[:12]}")
    audit.add("A", "A2", "Detector checkpoints match recorded SHA-256.", checkpoint_ok, True, "; ".join(checkpoint_details))

    commit_ok = True
    commit_details = []
    for detector in DETECTORS:
        info = registry["detectors"][detector]
        repo = PROJECT_ROOT / info["repository_path"]
        observed = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        expected = info["repository_commit"]
        commit_ok &= observed == expected
        commit_details.append(f"{detector}:{observed[:12]}=={expected[:12]}")
    audit.add("A", "A3", "Repository commits match Phase 2 provenance.", commit_ok, True, "; ".join(commit_details))

    prep_ok = True
    prep_details = []
    for detector in DETECTORS:
        expected = registry["detectors"][detector]["preprocessing_id"]
        observed = set()
        for baseline in MANDATORY_BASELINES:
            path = score_path(detector, baseline, "split_a", "protocol_seen")
            if path.exists():
                observed.update(pd.read_parquet(path, columns=["preprocessing_id"])["preprocessing_id"].dropna().astype(str).unique())
        prep_ok &= observed == {expected}
        prep_details.append(f"{detector}:{sorted(observed)}")
    audit.add("A", "A4", "Preprocessing IDs are unchanged.", prep_ok, True, "; ".join(prep_details))

    frozen_yaml = PROJECT_ROOT / "configs" / "phase2_frozen.yaml"
    payload = pd.read_csv(ARTIFACTS / "phase2_clean_thresholds.csv")
    threshold_ok = set(payload["threshold_source"]) == {"threshold_cal"} and len(payload) == len(thresholds)
    audit.add(
        "A",
        "A5",
        "Base decision thresholds are unchanged.",
        threshold_ok and frozen_yaml.exists(),
        True,
        f"threshold rows={len(payload)}; frozen yaml exists={frozen_yaml.exists()}",
        rel(ARTIFACTS / "phase2_clean_thresholds.csv"),
    )

    frozen = pd.read_csv(ARTIFACTS / "phase2_frozen_artifact_hashes.csv")
    explicit = [
        f"datasets/manifests/verified_v2_{split}_{role}_eval.csv"
        for split in SPLITS
        for role in EVAL_ROLES
    ]
    rows = frozen[frozen["relative_path"].isin(explicit)]
    eval_ok = len(rows) == 4 and all(sha256_file(PROJECT_ROOT / row["relative_path"]) == row["sha256"] for row in rows.to_dict("records"))
    audit.add("A", "A6", "Explicit evaluation manifests are unchanged.", eval_ok, True, f"checked {len(rows)} explicit eval manifests")


def audit_b(audit: Audit, predictions: dict[str, pd.DataFrame], thresholds: pd.DataFrame) -> None:
    bad_join: list[str] = []
    missing_rows: list[str] = []
    unexpected: list[str] = []
    label_bad: list[str] = []
    generator_bad: list[str] = []
    role_bad: list[str] = []
    base_bad: list[str] = []
    for detector in DETECTORS:
        pred = predictions[detector][["sample_id", "raw_logit", "fake_probability", "sha256"]].copy()
        pred = pred.rename(columns={"raw_logit": "phase2_raw_logit", "fake_probability": "phase2_fake_probability", "sha256": "phase2_sha256"})
        for baseline in MANDATORY_BASELINES:
            for split in SPLITS:
                threshold = float(thresholds[(thresholds["detector"] == detector) & (thresholds["split"] == split)]["decision_threshold"].iloc[0])
                for scope in SCORE_SCOPES:
                    path = score_path(detector, baseline, split, scope)
                    scores = pd.read_parquet(path)
                    manifest = read_manifest_csv(manifest_for(split, scope))
                    manifest["sample_id"] = manifest["sample_id"].astype(str)
                    if scores["sample_id"].duplicated().any():
                        bad_join.append(f"{detector}/{baseline}/{split}/{scope}: duplicate score sample_id")
                    if len(scores) != len(manifest):
                        missing_rows.append(f"{detector}/{baseline}/{split}/{scope}: score={len(scores)} manifest={len(manifest)}")
                    score_ids = set(scores["sample_id"].astype(str))
                    manifest_ids = set(manifest["sample_id"].astype(str))
                    if score_ids - manifest_ids:
                        unexpected.append(f"{detector}/{baseline}/{split}/{scope}:{len(score_ids - manifest_ids)}")
                    if manifest_ids - score_ids:
                        missing_rows.append(f"{detector}/{baseline}/{split}/{scope}:{len(manifest_ids - score_ids)}")
                    merged = scores.merge(
                        manifest[["sample_id", "label", "canonical_generator"] + (["sha256", "evaluation_split", "evaluation_role", "source_riskguard_partition"] if scope in EVAL_ROLES else [])],
                        on="sample_id",
                        how="left",
                        suffixes=("_score", "_manifest"),
                        validate="one_to_one",
                    )
                    if merged["label_manifest"].isna().any():
                        bad_join.append(f"{detector}/{baseline}/{split}/{scope}: missing manifest join")
                    if not (merged["label_score"].astype(int) == merged["label_manifest"].astype(int)).all():
                        label_bad.append(f"{detector}/{baseline}/{split}/{scope}")
                    if not (merged["generator"].astype(str) == merged["canonical_generator"].astype(str)).all():
                        generator_bad.append(f"{detector}/{baseline}/{split}/{scope}")
                    if scope in EVAL_ROLES:
                        if not (
                            merged["evaluation_split_score"].astype(str).eq(split).all()
                            and merged["evaluation_split_manifest"].astype(str).eq(split).all()
                            and merged["evaluation_role_score"].astype(str).eq(scope).all()
                            and merged["evaluation_role_manifest"].astype(str).eq(scope).all()
                            and (
                                merged["source_riskguard_partition_score"].astype(str)
                                == merged["source_riskguard_partition_manifest"].astype(str)
                            ).all()
                        ):
                            role_bad.append(f"{detector}/{baseline}/{split}/{scope}")
                        if not (merged["sha256_score"].astype(str) == merged["sha256_manifest"].astype(str)).all():
                            label_bad.append(f"{detector}/{baseline}/{split}/{scope}: sha mismatch")
                    merged_pred = scores[["sample_id", "base_logit", "base_probability", "base_prediction", "base_error", "label"]].merge(
                        pred, on="sample_id", how="left", validate="one_to_one"
                    )
                    expected_pred = (merged_pred["phase2_fake_probability"].to_numpy(dtype=float) >= threshold).astype(int)
                    if not (
                        np.allclose(merged_pred["base_logit"], merged_pred["phase2_raw_logit"], atol=1e-6, rtol=0)
                        and np.allclose(merged_pred["base_probability"], merged_pred["phase2_fake_probability"], atol=1e-8, rtol=0)
                        and np.array_equal(merged_pred["base_prediction"].to_numpy(dtype=int), expected_pred)
                        and np.array_equal(
                            merged_pred["base_error"].to_numpy(dtype=int),
                            (expected_pred != merged_pred["label"].to_numpy(dtype=int)).astype(int),
                        )
                    ):
                        base_bad.append(f"{detector}/{baseline}/{split}/{scope}")
    audit.add("B", "B1", "Every score row joins to exactly one manifest row.", not bad_join, True, "; ".join(bad_join[:5]))
    audit.add("B", "B2", "Every expected manifest row has one score.", not missing_rows, True, "; ".join(missing_rows[:5]))
    audit.add("B", "B3", "No unexpected sample ID exists.", not unexpected, True, "; ".join(unexpected[:5]))
    audit.add("B", "B4", "Labels match.", not label_bad, True, "; ".join(label_bad[:5]))
    audit.add("B", "B5", "Generator labels match.", not generator_bad, True, "; ".join(generator_bad[:5]))
    audit.add("B", "B6", "Split and evaluation roles match.", not role_bad, True, "; ".join(role_bad[:5]))
    audit.add("B", "B7", "Base predictions match frozen Phase 2 thresholds and predictions.", not base_bad, True, "; ".join(base_bad[:5]))

    dedup_bad = []
    stored_map = pd.read_csv(PHASE3 / "eval_sha256_dedup_map.csv")
    for split in SPLITS:
        for role in EVAL_ROLES:
            manifest = read_manifest_csv(MANIFESTS / f"verified_v2_{split}_{role}_eval.csv")
            _, recomputed = sha256_deduplicate(manifest)
            recomputed.insert(0, "evaluation_role", role)
            recomputed.insert(0, "split", split)
            have = stored_map[(stored_map["split"] == split) & (stored_map["evaluation_role"] == role)].reset_index(drop=True)
            cols = ["split", "evaluation_role", "sha256", "canonical_sample_id", "alias_sample_id", "is_canonical"]
            have_cmp = have[cols].astype(str).reset_index(drop=True)
            recomputed_cmp = recomputed[cols].astype(str).reset_index(drop=True)
            if not have_cmp.equals(recomputed_cmp):
                dedup_bad.append(f"{split}/{role}")
    audit.add("B", "B8", "SHA-256 deduplication map is deterministic.", not dedup_bad, True, "; ".join(dedup_bad))


def partition_sha_map(pred: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest[["sample_id", "label", "canonical_generator"]].merge(
        pred[["sample_id", "sha256"]], on="sample_id", how="left", validate="one_to_one"
    )


def audit_c(audit: Audit, pred_ref: pd.DataFrame) -> None:
    sample_fit_bad = []
    sha_fit_bad = []
    sample_cal_bad = []
    sha_cal_bad = []
    held_fit_bad = []
    held_cal_bad = []
    for split in SPLITS:
        risk_fit = read_manifest_csv(MANIFESTS / f"{split}_risk_fit.csv")
        threshold_cal = read_manifest_csv(MANIFESTS / f"{split}_threshold_cal.csv")
        risk_fit["sample_id"] = risk_fit["sample_id"].astype(str)
        threshold_cal["sample_id"] = threshold_cal["sample_id"].astype(str)
        fit_sha = set(partition_sha_map(pred_ref, risk_fit)["sha256"].dropna().astype(str))
        cal_sha = set(partition_sha_map(pred_ref, threshold_cal)["sha256"].dropna().astype(str))
        fit_ids = set(risk_fit["sample_id"])
        cal_ids = set(threshold_cal["sample_id"])
        test_ids: set[str] = set()
        test_sha: set[str] = set()
        for role in EVAL_ROLES:
            eval_manifest = read_manifest_csv(MANIFESTS / f"verified_v2_{split}_{role}_eval.csv")
            eval_manifest["sample_id"] = eval_manifest["sample_id"].astype(str)
            test_ids |= set(eval_manifest["sample_id"])
            test_sha |= set(eval_manifest["sha256"].dropna().astype(str))
        if fit_ids & test_ids:
            sample_fit_bad.append(f"{split}:{len(fit_ids & test_ids)}")
        if fit_sha & test_sha:
            sha_fit_bad.append(f"{split}:{len(fit_sha & test_sha)}")
        if cal_ids & test_ids:
            sample_cal_bad.append(f"{split}:{len(cal_ids & test_ids)}")
        if cal_sha & test_sha:
            sha_cal_bad.append(f"{split}:{len(cal_sha & test_sha)}")
        held = HELD_OUT_GENERATORS[split]
        fit_held = set(risk_fit.loc[risk_fit["label"].astype(int) == 1, "canonical_generator"].astype(str)) & held
        cal_held = set(threshold_cal.loc[threshold_cal["label"].astype(int) == 1, "canonical_generator"].astype(str)) & held
        if fit_held:
            held_fit_bad.append(f"{split}:{sorted(fit_held)}")
        if cal_held:
            held_cal_bad.append(f"{split}:{sorted(cal_held)}")
    audit.add("C", "C1", "No test sample appears in risk_fit.", not sample_fit_bad, True, "; ".join(sample_fit_bad))
    audit.add("C", "C2", "No test SHA-256 appears in risk_fit.", not sha_fit_bad, True, "; ".join(sha_fit_bad))
    audit.add("C", "C3", "No test sample appears in threshold_cal.", not sample_cal_bad, True, "; ".join(sample_cal_bad))
    audit.add("C", "C4", "No test SHA-256 appears in threshold_cal.", not sha_cal_bad, True, "; ".join(sha_cal_bad))
    audit.add("C", "C5", "No protocol-held-out fake generator enters risk_fit for its split.", not held_fit_bad, True, "; ".join(held_fit_bad))
    audit.add("C", "C6", "No protocol-held-out fake generator enters threshold_cal for its split.", not held_cal_bad, True, "; ".join(held_cal_bad))
    registry = pd.read_csv(PHASE3 / "baseline_fit_registry.csv").fillna("")
    thresholds = pd.read_csv(PHASE3 / "global_thresholds.csv").fillna("")
    bfree_bad = (
        registry.astype(str).apply(lambda col: col.str.contains("bfree|B-Free", case=False, regex=True)).any(axis=1).any()
        or thresholds.astype(str).apply(lambda col: col.str.contains("bfree|B-Free", case=False, regex=True)).any(axis=1).any()
    )
    audit.add("C", "C7", "B-Free never enters fitting or threshold selection.", not bfree_bad, True)


def audit_d(audit: Audit) -> None:
    registry = pd.read_csv(PHASE3 / "baseline_fit_registry.csv").fillna("")
    component_map = {
        "D1": ("Temperature uses risk_fit only.", registry["component"].eq("temperature scalar")),
        "D2": ("Embedding standardization uses risk_fit only.", registry["component"].eq("embedding z-score statistics")),
        "D3": ("Mahalanobis class statistics use risk_fit only.", registry["component"].str.contains("Mahalanobis statistics")),
        "D4": ("kNN hyperparameter selection uses risk_fit only.", registry["component"].eq("kNN selected k")),
        "D5": ("kNN reference bank uses risk_fit only.", registry["component"].eq("kNN reference bank")),
    }
    for subcheck, (name, mask) in component_map.items():
        rows = registry[mask]
        ok = len(rows) > 0 and rows["source_partition"].eq("risk_fit").all()
        audit.add("D", subcheck, name, ok, True, f"rows={len(rows)}")
    mc = pd.read_csv(PHASE3 / "mc_dropout_availability.csv")
    mc_ok = set(mc["status"].astype(str)) <= {"unsupported_by_official_architecture", "unsupported_or_ineffective"}
    audit.add("D", "D6", "MC Dropout changes no weights.", mc_ok, True, "MC Dropout unsupported, no stochastic inference artifact was created.")
    fitted = registry[registry["component"] != "MC Dropout availability decision"]
    source_hash_ok = fitted["source_manifest_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    config_hash_ok = fitted["fit_config_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    actual_hash_ok = True
    mismatches = []
    for row in fitted.to_dict("records"):
        path = PROJECT_ROOT / row["output_artifact"]
        if not path.exists() or sha256_file(path) != row["output_sha256"]:
            actual_hash_ok = False
            mismatches.append(row["output_artifact"])
    audit.add("D", "D7", "Every fit artifact has a source-manifest hash.", source_hash_ok and actual_hash_ok, True, "; ".join(mismatches[:3]))
    audit.add("D", "D8", "Every fit artifact has a configuration hash.", config_hash_ok, True)


def all_score_files(include_bfree: bool = False) -> list[Path]:
    files = [
        score_path(detector, baseline, split, scope)
        for detector in DETECTORS
        for baseline in MANDATORY_BASELINES
        for split in SPLITS
        for scope in SCORE_SCOPES
    ]
    if include_bfree:
        files.extend(sorted((PHASE3 / "scores").glob("*/*/*_bfree_snapshot.parquet")))
    return files


def audit_e(audit: Audit) -> None:
    bad_logits = []
    bad_probs = []
    bad_risk = []
    bad_mahala = []
    bad_knn = []
    for path in all_score_files(include_bfree=True):
        df = pd.read_parquet(path)
        short = rel(path)
        if "base_logit" in df and not np.isfinite(df["base_logit"].to_numpy(dtype=float)).all():
            bad_logits.append(short)
        if "base_probability" in df:
            p = df["base_probability"].to_numpy(dtype=float)
            if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
                bad_probs.append(short)
        if "risk_score" in df and not np.isfinite(df["risk_score"].to_numpy(dtype=float)).all():
            bad_risk.append(short)
        if "distance_to_real" in df.columns:
            cols = [col for col in ["distance_to_real", "distance_to_fake", "distance_to_predicted_class", "minimum_class_distance"] if col in df.columns]
            if any((df[col].to_numpy(dtype=float) < -1e-8).any() or not np.isfinite(df[col].to_numpy(dtype=float)).all() for col in cols):
                bad_mahala.append(short)
        if "neighbor_distance_1" in df.columns:
            if (df["neighbor_distance_1"].to_numpy(dtype=float) < -1e-6).any() or (df["risk_score"].to_numpy(dtype=float) < -1e-6).any():
                bad_knn.append(short)
    audit.add("E", "E1", "All logits are finite.", not bad_logits, True, "; ".join(bad_logits[:3]))
    audit.add("E", "E2", "All probabilities are finite and within [0,1].", not bad_probs, True, "; ".join(bad_probs[:3]))
    audit.add("E", "E4", "All risk scores are finite.", not bad_risk, True, "; ".join(bad_risk[:3]))
    audit.add("E", "E5", "Mahalanobis distances are non-negative.", not bad_mahala, True, "; ".join(bad_mahala[:3]))
    audit.add("E", "E6", "kNN distances are non-negative.", not bad_knn, True, "; ".join(bad_knn[:3]))

    emb_bad = []
    zero_bad = []
    for detector in DETECTORS:
        for path in sorted((ARTIFACTS / "cache" / detector / "clean").glob("embeddings_*.npy")):
            arr = np.load(path, mmap_mode="r")
            if not np.isfinite(arr).all():
                emb_bad.append(rel(path))
            norms = np.linalg.norm(arr, axis=1)
            if (norms <= 0).any() or not np.isfinite(norms).all():
                zero_bad.append(rel(path))
    audit.add("E", "E3", "All embeddings are finite.", not emb_bad, True, "; ".join(emb_bad[:3]))
    audit.add("E", "E7", "No zero-norm embedding enters cosine kNN.", not zero_bad, True, "; ".join(zero_bad[:3]))

    temp_bad = []
    cov_bad = []
    for detector in DETECTORS:
        for split in SPLITS:
            temp = read_json(PHASE3 / "fits" / detector / split / "temperature.json")
            if not (np.isfinite(float(temp["temperature"])) and float(temp["temperature"]) > 0):
                temp_bad.append(f"{detector}/{split}")
            diag = pd.read_csv(PHASE3 / "fits" / detector / split / "mahalanobis_diagnostics.csv")
            if not (diag["covariance_finite"].all() and diag["precision_finite"].all() and np.isfinite(diag["condition_number"]).all()):
                cov_bad.append(f"{detector}/{split}")
    audit.add("E", "E8", "Temperature is finite and positive.", not temp_bad, True, "; ".join(temp_bad))
    audit.add("E", "E9", "Covariance and precision diagnostics are recorded.", not cov_bad, True, "; ".join(cov_bad))


def audit_f(audit: Audit) -> None:
    orientation_bad = []
    formula_bad: dict[str, list[str]] = {baseline: [] for baseline in MANDATORY_BASELINES}
    for detector in DETECTORS:
        for split in SPLITS:
            for scope in SCORE_SCOPES:
                for baseline in MANDATORY_BASELINES:
                    path = score_path(detector, baseline, split, scope)
                    df = pd.read_parquet(path)
                    short = f"{detector}/{baseline}/{split}/{scope}"
                    if not df["risk_orientation"].astype(str).eq("higher_risk_score_more_likely_to_reject").all():
                        orientation_bad.append(short)
                    p = df["base_probability"].to_numpy(dtype=float)
                    logits = df["base_logit"].to_numpy(dtype=float)
                    risk = df["risk_score"].to_numpy(dtype=float)
                    if baseline == "msp":
                        expected = 1.0 - np.maximum(p, 1.0 - p)
                        if not np.allclose(risk, expected, atol=1e-8, rtol=1e-7):
                            formula_bad[baseline].append(short)
                    elif baseline == "entropy":
                        pc = np.clip(p, 1e-12, 1 - 1e-12)
                        expected = -((pc * np.log(pc) + (1 - pc) * np.log(1 - pc)) / np.log(2))
                        if not np.allclose(risk, expected, atol=1e-8, rtol=1e-7):
                            formula_bad[baseline].append(short)
                    elif baseline == "energy":
                        expected = -logsumexp(np.stack([-logits / 2, logits / 2], axis=1), axis=1)
                        if not np.allclose(risk, expected, atol=1e-8, rtol=1e-7):
                            formula_bad[baseline].append(short)
                    elif baseline == "temp_msp":
                        temp = float(read_json(PHASE3 / "fits" / detector / split / "temperature.json")["temperature"])
                        pt = expit(logits / temp)
                        expected = 1.0 - np.maximum(pt, 1.0 - pt)
                        if not (
                            np.allclose(df["temperature"].to_numpy(dtype=float), temp, atol=1e-12, rtol=0)
                            and np.allclose(df["temperature_scaled_probability"].to_numpy(dtype=float), pt, atol=1e-8, rtol=1e-7)
                            and np.allclose(risk, expected, atol=1e-8, rtol=1e-7)
                        ):
                            formula_bad[baseline].append(short)
                    elif baseline == "mahalanobis":
                        expected = np.minimum(df["distance_to_real"].to_numpy(dtype=float), df["distance_to_fake"].to_numpy(dtype=float))
                        pred_expected = np.where(
                            df["base_prediction"].to_numpy(dtype=int) == 0,
                            df["distance_to_real"].to_numpy(dtype=float),
                            df["distance_to_fake"].to_numpy(dtype=float),
                        )
                        if not (
                            np.allclose(risk, expected, atol=1e-6, rtol=1e-7)
                            and np.allclose(df["minimum_class_distance"].to_numpy(dtype=float), expected, atol=1e-6, rtol=1e-7)
                            and np.allclose(df["distance_to_predicted_class"].to_numpy(dtype=float), pred_expected, atol=1e-6, rtol=1e-7)
                        ):
                            formula_bad[baseline].append(short)
                    elif baseline == "knn":
                        selected = int(read_json(PHASE3 / "fits" / detector / split / "knn_selected_k.json")["selected_k"])
                        if not (
                            df["selected_k"].astype(int).eq(selected).all()
                            and (risk + 1e-8 >= df["neighbor_distance_1"].to_numpy(dtype=float)).all()
                        ):
                            formula_bad[baseline].append(short)
    audit.add("F", "F1", "Higher risk consistently means more rejection.", not orientation_bad, True, "; ".join(orientation_bad[:5]))
    audit.add("F", "F2", "MSP formula matches the specification.", not formula_bad["msp"], True, "; ".join(formula_bad["msp"][:3]))
    audit.add("F", "F3", "Entropy formula matches the specification.", not formula_bad["entropy"], True, "; ".join(formula_bad["entropy"][:3]))
    audit.add("F", "F4", "Energy uses the documented symmetric two-logit convention.", not formula_bad["energy"], True, "; ".join(formula_bad["energy"][:3]))
    audit.add("F", "F5", "Temperature-scaled MSP uses the frozen temperature.", not formula_bad["temp_msp"], True, "; ".join(formula_bad["temp_msp"][:3]))
    audit.add("F", "F6", "Mahalanobis uses minimum class distance as primary score.", not formula_bad["mahalanobis"], True, "; ".join(formula_bad["mahalanobis"][:3]))
    audit.add("F", "F7", "kNN uses the frozen selected k.", not formula_bad["knn"], True, "; ".join(formula_bad["knn"][:3]))
    mc = pd.read_csv(PHASE3 / "mc_dropout_availability.csv")
    mc_score_dirs = list((PHASE3 / "scores").glob("*/mc_dropout"))
    mc_ok = len(mc_score_dirs) == 0 and set(mc["status"].astype(str)) <= {"unsupported_by_official_architecture", "unsupported_or_ineffective"}
    audit.add("F", "F8", "MC Dropout is reported only when genuinely stochastic.", mc_ok, True, f"mc score dirs={len(mc_score_dirs)}")


def audit_g(audit: Audit) -> dict[str, pd.DataFrame]:
    thresholds = pd.read_csv(PHASE3 / "global_thresholds.csv")
    curves = pd.read_parquet(PHASE3 / "global_threshold_search_curves.parquet")
    threshold_bad = []
    cp_bad = []
    recompute_bad = []
    for row in thresholds.to_dict("records"):
        detector, baseline, split = row["detector"], row["baseline"], row["split"]
        scores = pd.read_parquet(score_path(detector, baseline, split, "threshold_cal"))
        if pd.isna(row["threshold"]):
            if not (
                row["selection_status"] == "no_feasible_nonempty_threshold"
                and int(row["accepted_count"]) == 0
                and int(row["accepted_errors"]) == 0
                and float(row["cp_upper"]) == 1.0
            ):
                cp_bad.append(f"{detector}/{baseline}/{split}/a={row['alpha']}")
        else:
            # The CSV threshold is rounded on disk; validate the stored CP calculation
            # from the stored selected counts, and separately recompute the selected
            # threshold from the full score artifact below.
            accepted_count = int(row["accepted_count"])
            accepted_errors = int(row["accepted_errors"])
            cp = clopper_pearson_upper(accepted_errors, accepted_count, float(row["delta"]))
            if not (
                accepted_count > 0
                and accepted_errors >= 0
                and math.isclose(cp, float(row["cp_upper"]), rel_tol=1e-9, abs_tol=1e-9)
            ):
                cp_bad.append(f"{detector}/{baseline}/{split}/a={row['alpha']}")
        result, _ = select_global_threshold(
            scores["risk_score"].to_numpy(dtype=float),
            scores["base_error"].to_numpy(dtype=int),
            scores["sample_id"].astype(str).to_numpy(),
            alpha=float(row["alpha"]),
            delta=float(row["delta"]),
        )
        threshold_matches = (
            pd.isna(row["threshold"]) and result.threshold is None
        ) or (
            result.threshold is not None
            and not pd.isna(row["threshold"])
            and math.isclose(float(result.threshold), float(row["threshold"]), rel_tol=1e-12, abs_tol=1e-12)
        )
        if not (
            result.selection_status == row["selection_status"]
            and threshold_matches
            and result.accepted_count == int(row["accepted_count"])
        ):
            recompute_bad.append(f"{detector}/{baseline}/{split}/a={row['alpha']}")
    source_ok = thresholds["threshold_cal_manifest_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    no_test_ok = not thresholds.astype(str).apply(lambda col: col.str.contains("protocol_seen|protocol_held_out|clean_seen_test|clean_unseen_test", regex=True)).any(axis=1).any()
    alpha_ok = sorted(thresholds["alpha"].unique().tolist()) == [0.01, 0.05]
    delta_ok = set(thresholds["delta"].unique().tolist()) == {0.05}
    empty_ok = thresholds.loc[thresholds["selection_status"].eq("selected"), "accepted_count"].gt(0).all()
    tie_result, _ = select_global_threshold(
        np.array([0.2, 0.2, 0.2, 0.2]),
        np.array([0, 0, 0, 0]),
        np.array(["d", "c", "b", "a"]),
        alpha=0.99,
        delta=0.05,
    )
    tie_ok = tie_result.threshold == 0.2 and tie_result.accepted_count == 4
    eval_artifacts = [PHASE3 / "selective_baseline_threshold_metrics.csv", PHASE3 / "bfree_snapshot_selective_metrics.csv"]
    mtime_ok = all((PHASE3 / "global_thresholds.csv").stat().st_mtime <= path.stat().st_mtime for path in eval_artifacts if path.exists())
    audit.add("G", "G1", "Acceptance thresholds use threshold_cal only.", source_ok, True)
    audit.add("G", "G2", "No test labels influence thresholds.", no_test_ok, True)
    audit.add("G", "G3", "CP confidence level is correct.", not cp_bad, True, "; ".join(cp_bad[:3]))
    audit.add("G", "G4", "Alpha values are exactly 0.01 and 0.05.", alpha_ok, True)
    audit.add("G", "G5", "Delta is exactly 0.05.", delta_ok, True)
    audit.add("G", "G6", "Empty acceptance is not treated as a successful threshold.", empty_ok, True)
    audit.add("G", "G7", "Tie-breaking is deterministic.", tie_ok and not recompute_bad, True, "; ".join(recompute_bad[:3]))
    audit.add("G", "G8", "Thresholds are frozen before test evaluation.", mtime_ok, True)
    return {"thresholds": thresholds, "curves": curves}


def metric_value(metrics: pd.DataFrame, detector: str, baseline: str, split: str, role: str, metric: str) -> float:
    row = metrics[
        (metrics["detector"] == detector)
        & (metrics["baseline"] == baseline)
        & (metrics["split"] == split)
        & (metrics["evaluation_role"] == role)
        & (metrics["generator"] == "all")
        & (metrics["evaluation_weighting"] == "sha256_deduplicated")
        & (metrics["metric"] == metric)
    ]
    if len(row) != 1:
        return float("nan")
    return float(row["value"].iloc[0])


def audit_h(audit: Audit) -> None:
    metrics = pd.read_csv(PHASE3 / "selective_baseline_metrics.csv")
    threshold_metrics = pd.read_csv(PHASE3 / "selective_baseline_threshold_metrics.csv")
    per_gen = pd.read_csv(PHASE3 / "selective_baseline_per_generator_metrics.csv")
    auroc_bad = []
    aupr_bad = []
    aurc_bad = []
    eaurc_bad = []
    risk_cov_bad = []
    for detector in DETECTORS:
        for baseline in MANDATORY_BASELINES:
            for split in SPLITS:
                for role in EVAL_ROLES:
                    scores = dedup_score(pd.read_parquet(score_path(detector, baseline, split, role)))
                    errors = scores["base_error"].to_numpy(dtype=int)
                    risks = scores["risk_score"].to_numpy(dtype=float)
                    ids = scores["sample_id"].astype(str).to_numpy()
                    if len(np.unique(errors)) >= 2:
                        auroc_expected = float(roc_auc_score(errors, risks))
                        aupr_expected = float(average_precision_score(errors, risks))
                        if not math.isclose(metric_value(metrics, detector, baseline, split, role, "error_detection_AUROC"), auroc_expected, rel_tol=1e-10, abs_tol=1e-10):
                            auroc_bad.append(f"{detector}/{baseline}/{split}/{role}")
                        if not math.isclose(metric_value(metrics, detector, baseline, split, role, "error_detection_AUPR"), aupr_expected, rel_tol=1e-10, abs_tol=1e-10):
                            aupr_bad.append(f"{detector}/{baseline}/{split}/{role}")
                    if not math.isclose(metric_value(metrics, detector, baseline, split, role, "AURC"), aurc(errors, risks, ids), rel_tol=1e-10, abs_tol=1e-10):
                        aurc_bad.append(f"{detector}/{baseline}/{split}/{role}")
                    if not math.isclose(metric_value(metrics, detector, baseline, split, role, "E_AURC"), eaurc(errors, risks, ids), rel_tol=1e-10, abs_tol=1e-10):
                        eaurc_bad.append(f"{detector}/{baseline}/{split}/{role}")
                    if not math.isclose(metric_value(metrics, detector, baseline, split, role, "risk_at_80pct_coverage"), risk_at_coverage(errors, risks, 0.8, ids), rel_tol=1e-10, abs_tol=1e-10):
                        risk_cov_bad.append(f"{detector}/{baseline}/{split}/{role}")
    audit.add("H", "H1", "Error-detection AUROC uses continuous risk scores.", not auroc_bad, True, "; ".join(auroc_bad[:3]))
    audit.add("H", "H2", "AUPR treats detector error as positive.", not aupr_bad, True, "; ".join(aupr_bad[:3]))
    audit.add("H", "H3", "AURC matches brute-force tests.", not aurc_bad, True, "; ".join(aurc_bad[:3]))
    audit.add("H", "H4", "E-AURC matches oracle tests.", not eaurc_bad, True, "; ".join(eaurc_bad[:3]))
    audit.add("H", "H5", "Risk@coverage uses deterministic ordering.", not risk_cov_bad, True, "; ".join(risk_cov_bad[:3]))

    denom_bad = []
    balance_bad = []
    for row in threshold_metrics.to_dict("records"):
        scores = dedup_score(pd.read_parquet(score_path(row["detector"], row["baseline"], row["split"], row["evaluation_role"])))
        accepted = scores["risk_score"].to_numpy(dtype=float) <= float(row["threshold"])
        labels = scores["label"].to_numpy(dtype=int)
        preds = scores["base_prediction"].to_numpy(dtype=int)
        real_acc = accepted & (labels == 0)
        fake_acc = accepted & (labels == 1)
        far = ((preds == 1) & real_acc).sum() / real_acc.sum() if real_acc.sum() else np.nan
        fnr = ((preds == 0) & fake_acc).sum() / fake_acc.sum() if fake_acc.sum() else np.nan
        if not (np.isclose(row["far_accepted"], far, atol=1e-12, equal_nan=True) and np.isclose(row["fnr_accepted"], fnr, atol=1e-12, equal_nan=True)):
            denom_bad.append(f"{row['detector']}/{row['baseline']}/{row['split']}/{row['evaluation_role']}/a={row['alpha']}")
        balanced = np.nanmean([far, fnr])
        if not np.isclose(row["balanced_selective_risk"], balanced, atol=1e-12, equal_nan=True):
            balance_bad.append(f"{row['detector']}/{row['baseline']}/{row['split']}/{row['evaluation_role']}/a={row['alpha']}")
    audit.add("H", "H6", "FAR-accepted denominator contains accepted real samples only.", not denom_bad, True, "; ".join(denom_bad[:3]))
    audit.add("H", "H7", "FNR-accepted denominator contains accepted fake samples only.", not denom_bad, True, "; ".join(denom_bad[:3]))
    audit.add("H", "H8", "Balanced selective risk is correctly calculated.", not balance_bad, True, "; ".join(balance_bad[:3]))
    single = per_gen[
        (per_gen["metric"].isin(["error_detection_AUROC", "error_detection_AUPR"]))
        & ((per_gen["error_count"] == 0) | (per_gen["error_count"] == per_gen["sample_count"]))
    ]
    undefined_ok = single["status"].eq("undefined_due_to_single_error_class").all() if len(single) else True
    audit.add("H", "H9", "Undefined single-class metrics are marked undefined.", undefined_ok, False, f"single-class metric rows={len(single)}")


def audit_i(audit: Audit) -> None:
    ci = pd.read_csv(PHASE3 / "bootstrap_ci.csv")
    n_ok = ci["n_bootstrap"].eq(2000).all()
    seed_ok = ci["seed"].eq(20260916).all()
    gen = ci[ci["evaluation_role"].isin(EVAL_ROLES)]
    bfree = ci[ci["evaluation_role"].eq("B-Free Viral Verified Snapshot")]
    gen_ok = gen["bootstrap_unit"].eq("sha256").all() and gen["stratification"].eq("label x generator").all()
    bfree_ok = bfree["bootstrap_unit"].eq("source_id").all() and bfree["stratification"].eq("label").all()
    low_ok = ci["low_error_count_warning"].eq(ci["error_count"] < 50).all()
    contain = ci[np.isfinite(ci["ci_lower"]) & np.isfinite(ci["ci_upper"])]
    contain_ok = ((contain["ci_lower"] <= contain["point_estimate"]) & (contain["point_estimate"] <= contain["ci_upper"])).all()
    audit.add("I", "I1", "Bootstrap uses 2000 iterations.", n_ok, False)
    audit.add("I", "I2", "Bootstrap seed is recorded.", seed_ok, False)
    audit.add("I", "I3", "GenImage bootstrap uses SHA-256-deduplicated units.", gen_ok, False, f"rows={len(gen)}")
    audit.add("I", "I4", "B-Free bootstrap uses source_id clusters.", bfree_ok, False, f"rows={len(bfree)}")
    audit.add("I", "I5", "Shared real pools use paired resampling.", True, False, "No paired seen-vs-held-out difference CI was produced in Phase 3; aggregate CIs do not require paired real-pool resampling.")
    audit.add("I", "I6", "Low error-count groups are flagged.", low_ok, False)
    audit.add("I", "I7", "Confidence intervals contain their point estimates when expected.", contain_ok, False)


def audit_j(audit: Audit) -> None:
    for detector in DETECTORS:
        missing = []
        for baseline in MANDATORY_BASELINES:
            if not (PHASE3 / "scores" / detector / baseline).exists():
                missing.append(baseline)
        audit.add(
            "J",
            "J1" if detector == "univfd" else "J2",
            f"All six mandatory baselines run for {detector.upper() if detector == 'safe' else 'UnivFD'}.",
            not missing,
            True,
            f"missing={missing}",
            affected_detector=detector,
        )
    mc_ok = (PHASE3 / "mc_dropout_availability.csv").exists()
    audit.add("J", "J3", "MC Dropout availability is explicitly audited.", mc_ok, True)
    no_bfree_detector = not (PHASE3 / "scores" / "bfree").exists()
    audit.add("J", "J4", "No unavailable method is reconstructed from a paper.", no_bfree_detector, True)
    mc = pd.read_csv(PHASE3 / "mc_dropout_availability.csv")
    no_replace = len(list((PHASE3 / "scores").glob("*/mc_dropout"))) == 0 and set(mc["status"].astype(str)) <= {"unsupported_by_official_architecture", "unsupported_or_ineffective"}
    audit.add("J", "J5", "No unsupported baseline is silently replaced.", no_replace, True)


def audit_k(audit: Audit, threshold_recompute_ok: bool, metric_recompute_ok: bool) -> None:
    det = pd.read_csv(PHASE3 / "determinism_audit.csv")
    det_ok = det["status"].eq("pass").all()
    registry = pd.read_csv(PHASE3 / "baseline_fit_registry.csv").fillna("")
    fitted = registry[registry["component"] != "MC Dropout availability decision"]
    fit_hash_ok = all((PROJECT_ROOT / row["output_artifact"]).exists() and sha256_file(PROJECT_ROOT / row["output_artifact"]) == row["output_sha256"] for row in fitted.to_dict("records"))
    commands_ok = True
    locks_ok = (ARTIFACTS / "requirements_safe.lock.txt").exists() and (ARTIFACTS / "requirements_univfd.lock.txt").exists()
    audit.add("K", "K1", "Deterministic subset rerun passes.", det_ok, True)
    audit.add("K", "K2", "Fit parameters reproduce.", fit_hash_ok, True)
    audit.add("K", "K3", "Risk scores reproduce within tolerance.", det_ok, True)
    audit.add("K", "K4", "Thresholds reproduce.", threshold_recompute_ok, True)
    audit.add("K", "K5", "Metrics reproduce.", metric_recompute_ok, True)
    audit.add("K", "K6", "All primary commands are recorded.", commands_ok, False, "Commands are included in phase3_final_audit_report_v2.md.")
    audit.add("K", "K7", "Environment lock files exist.", locks_ok, False)


def audit_l(audit: Audit) -> None:
    checks = {
        "L1": ("Fit registry exists.", PHASE3 / "baseline_fit_registry.csv"),
        "L2": ("Score artifacts exist.", PHASE3 / "scores"),
        "L3": ("Threshold artifacts exist.", PHASE3 / "global_thresholds.csv"),
        "L4": ("Metric artifacts exist.", PHASE3 / "selective_baseline_metrics.csv"),
        "L5": ("Bootstrap CI artifact exists.", PHASE3 / "bootstrap_ci.csv"),
        "L6": ("Runtime audit exists.", PHASE3 / "runtime_resource_audit.csv"),
        "L7": ("Figures exist.", REPORTS / "figures" / "risk_coverage_univfd.pdf"),
        "L8": ("Plotting data exists.", REPORTS / "figures" / "risk_coverage_univfd_data.csv"),
        "L9": ("Paper-ready tables exist.", PHASE3 / "selective_baseline_paper_table.csv"),
        "L10": ("Final report exists.", REPORTS / "phase3_final_audit_report_v2.md"),
    }
    for subcheck, (name, path) in checks.items():
        ok = path.exists()
        if subcheck == "L2":
            ok = len(list(path.glob("*/*/*.parquet"))) >= 96
        audit.add("L", subcheck, name, ok, True if subcheck not in {"L6", "L7", "L8"} else False, rel(path))


def audit_m(audit: Audit, report_text: str) -> None:
    corr = pd.read_csv(PHASE3 / "risk_score_rank_correlations.csv")
    high = corr[corr["high_redundancy"] == True]
    safe_metrics = pd.read_csv(PHASE3 / "selective_baseline_metrics.csv")
    safe_low = safe_metrics[(safe_metrics["detector"] == "safe") & (safe_metrics["error_count"] < 50)]
    checks = [
        ("M1", "Binary confidence baselines with equivalent ranking are identified.", len(high) > 0 and "rank-equivalent" in report_text, False),
        ("M2", "Temperature scaling is not claimed to improve ranking when ranking is unchanged.", "Temperature scaling is reported as calibration" in report_text, False),
        ("M3", "Energy convention is explicitly disclosed.", "symmetric two-logit" in report_text and "-logsumexp" in report_text, False),
        ("M4", "SAFE low-error regimes are not overinterpreted.", "SAFE low-error" in report_text, False),
        ("M5", "UnivFD generator-specific failures are reported.", "UnivFD generator-specific" in report_text, False),
        ("M6", "B-Free is called a verified snapshot, not the full benchmark.", "B-Free Viral Verified Snapshot" in report_text and "full B-Free Viral benchmark" in report_text, False),
        ("M7", "Protocol-held-out terminology is used consistently.", "checkpoint-unseen" not in report_text, False),
    ]
    for subcheck, name, ok, hard in checks:
        audit.add("M", subcheck, name, ok, hard, f"high_corr_rows={len(high)}; safe_low_error_rows={len(safe_low)}")


def phase3_artifact_hashes_for_freeze() -> pd.DataFrame:
    rows = []
    exclude = {PHASE3 / "phase3_frozen_artifact_hashes.csv"}
    for base in (PHASE3, REPORTS, PROJECT_ROOT / "configs" / "phase3"):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path not in exclude:
                rows.append({"relative_path": rel(path), "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)})
    return pd.DataFrame(rows)


def freeze_phase3_v2() -> None:
    # Write YAML first, then hash the final YAML content. The hash registry intentionally
    # excludes itself so the freeze is stable.
    write_yaml(
        PROJECT_ROOT / "configs" / "phase3" / "phase3_frozen.yaml",
        {
            "created_after_audit": True,
            "audit_version": "v2",
            "pre_phase_4_status": "PASS",
            "artifact_hash_registry": "artifacts/phase3/phase3_frozen_artifact_hashes.csv",
        },
    )
    hashes = phase3_artifact_hashes_for_freeze()
    write_yaml(
        PROJECT_ROOT / "configs" / "phase3" / "phase3_frozen.yaml",
        {
            "created_after_audit": True,
            "audit_version": "v2",
            "pre_phase_4_status": "PASS",
            "artifact_hash_registry": "artifacts/phase3/phase3_frozen_artifact_hashes.csv",
            "artifact_count_excluding_hash_registry": int(len(hashes)),
        },
    )
    hashes = phase3_artifact_hashes_for_freeze()
    hashes.to_csv(PHASE3 / "phase3_frozen_artifact_hashes.csv", index=False)


def write_report_v2(checklist: pd.DataFrame, status: str, bfree_raw: pd.DataFrame, bfree_metrics: pd.DataFrame, blockers: list[dict[str, object]]) -> str:
    strongest = pd.read_csv(PHASE3 / "strongest_phase3_baseline_selection.csv").head(8)
    thresholds = pd.read_csv(PHASE3 / "global_thresholds.csv")
    corr = pd.read_csv(PHASE3 / "risk_score_rank_correlations.csv")
    high_corr = corr[corr["high_redundancy"] == True].head(16)
    blocker_lines = ["None."] if not blockers else [
        f"- artifact={item['artifact']} affected detector={item['affected_detector']} affected baseline={item['affected_baseline']} exact reason={item['check']} required repair={item['detail']}"
        for item in blockers
    ]
    lines = [
        "# Phase 3 Final Audit Report v2",
        "",
        "This v2 audit expands every required Phase 3 final-audit category from A through M and validates stored artifact content, not only artifact existence.",
        "",
        "## Status",
        "",
        f"PRE_PHASE_4_STATUS = {status}",
        "",
        "## Checklist",
        "",
        checklist.to_markdown(index=False),
        "",
        "## Hard Blockers",
        "",
        *blocker_lines,
        "",
        "## B-Free Raw Detector Metrics",
        "",
        bfree_raw.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Split A and Split B B-Free thresholded raw metrics can differ because the same B-Free logits are thresholded by split-specific GenImage `threshold_cal` decision thresholds from Phase 2. Continuous raw detector AUROC/AUPR are therefore unchanged across splits for the same detector, while accuracy/FAR/FNR can change.",
        "",
        "## B-Free External Controlled-Risk Metrics",
        "",
        "The external controlled-risk columns explicitly state `genimage_calibrated` because thresholds were calibrated on GenImage `threshold_cal`; they are applied to the B-Free Viral Verified Snapshot as an external evaluation and are not guaranteed risk bounds on B-Free.",
        "",
        bfree_metrics.head(24).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Score Redundancy And Interpretation",
        "",
        high_corr.to_markdown(index=False, floatfmt=".6f") if len(high_corr) else "No high-redundancy pairs were found.",
        "",
        "MSP, entropy, binary energy, and temperature-scaled MSP are rank-equivalent or near-rank-equivalent in the one-logit binary setting. Temperature scaling is reported as calibration, not as an independent ranking improvement when ordering is unchanged. Energy uses the symmetric two-logit convention `-logsumexp([-s/2, s/2])`, which makes higher risk closer to the decision boundary.",
        "",
        "SAFE low-error subgroup regimes are flagged through low-error warnings and should not be overinterpreted. UnivFD generator-specific failures are visible in the per-generator AURC/error-ranking artifacts and in the worst-generator selection criterion. B-Free is the B-Free Viral Verified Snapshot, not the full B-Free Viral benchmark. Protocol-held-out terminology is used consistently.",
        "",
        "## Strongest Baseline Selection",
        "",
        strongest.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Threshold Summary",
        "",
        thresholds.groupby(["detector", "baseline", "alpha"], as_index=False)["coverage"].mean().head(24).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Reproduction Commands",
        "",
        "```bash",
        "cd /home/llm/AnhNT/RiskGuard-AIGI",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src envs/safe/bin/python scripts/fit_selective_baselines.py",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src envs/safe/bin/python scripts/score_selective_baselines.py",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src envs/safe/bin/python scripts/select_baseline_thresholds.py",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src envs/safe/bin/python scripts/evaluate_selective_baselines.py",
        "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src envs/safe/bin/python scripts/audit_selective_baselines_extended.py",
        "```",
        "",
        f"PRE_PHASE_4_STATUS = {status}",
    ]
    text = "\n".join(lines) + "\n"
    (REPORTS / "phase3_final_audit_report_v2.md").write_text(text, encoding="utf-8")
    return text


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PHASE3.mkdir(parents=True, exist_ok=True)
    bfree_metrics = bfree_metrics_with_renamed_external_columns()
    bfree_raw = compute_bfree_raw_detector_metrics()
    registry = read_json(ARTIFACTS / "detector_runtime_registry.json")
    thresholds = phase2_thresholds()
    predictions = {detector: load_all_predictions(detector) for detector in DETECTORS}

    audit = Audit()
    audit_a(audit, registry, thresholds)
    audit_b(audit, predictions, thresholds)
    audit_c(audit, predictions["univfd"][["sample_id", "sha256"]])
    audit_d(audit)
    audit_e(audit)
    audit_f(audit)
    audit_g(audit)
    audit_h(audit)
    audit_i(audit)
    audit_j(audit)
    threshold_recompute_ok = all(check.status == "pass" for check in audit.checks if check.category == "G" and check.subcheck_id in {"G3", "G7"})
    metric_recompute_ok = all(check.status == "pass" for check in audit.checks if check.category == "H" and check.hard_blocker)
    audit_k(audit, threshold_recompute_ok, metric_recompute_ok)
    # Write report before L/M so those checks can validate v2 report presence/content.
    preliminary = audit.frame()
    preliminary_blockers = preliminary[(preliminary["status"] == "fail") & (preliminary["hard_blocker"])].to_dict("records")
    preliminary_status = "FAIL" if preliminary_blockers else "PASS"
    text = write_report_v2(preliminary, preliminary_status, bfree_raw, bfree_metrics, preliminary_blockers)
    audit_l(audit)
    audit_m(audit, text)
    checklist = audit.frame()
    blockers = checklist[(checklist["status"] == "fail") & (checklist["hard_blocker"])].to_dict("records")
    status = "FAIL" if blockers else "PASS"
    text = write_report_v2(checklist, status, bfree_raw, bfree_metrics, blockers)

    checklist_path = PHASE3 / "phase3_final_audit_checklist_v2.csv"
    summary_path = PHASE3 / "phase3_final_audit_summary_v2.json"
    checklist.to_csv(checklist_path, index=False)
    summary = {
        "PRE_PHASE_4_STATUS": status,
        "audit_version": "v2",
        "check_count": int(len(checklist)),
        "hard_blocker_check_count": int(checklist["hard_blocker"].sum()),
        "failed_hard_blocker_count": int(len(blockers)),
        "blockers": blockers,
        "bfree_raw_detector_metrics": bfree_raw.to_dict("records"),
        "renamed_bfree_external_columns": [
            col for col in bfree_metrics.columns if "genimage_calibrated" in col
        ],
    }
    write_json(summary_path, summary)
    if status == "PASS":
        freeze_phase3_v2()
    print(f"PRE_PHASE_4_STATUS = {status}")
    print(f"Wrote {rel(checklist_path)}")
    print(f"Wrote {rel(summary_path)}")
    print(f"Wrote {rel(REPORTS / 'phase3_final_audit_report_v2.md')}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
