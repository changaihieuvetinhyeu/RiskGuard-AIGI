#!/usr/bin/env python3
"""Repair cosine-kNN with SHA-grouped CV and a deduplicated bank."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import run_disjoint_pipeline as clean
from selective_detection.selective_baselines import Phase2Cache, exact_knn_neighbors


ART = clean.ART
REP = clean.REP
REPAIR = ART / "baselines" / "knn_sha_group_repair"
SENS = REPAIR / "sensitivity"
ARCHIVE = ART / "baselines" / "pre_sha_group_repair"
CANDIDATE_K = (1, 5, 10, 20, 50)
FINAL_REFERENCE_CONFIGURATION = "one_row_per_sha256"

# Frozen before sensitivity results are observed.
MATERIALITY = {
    "selected_k_must_match": True,
    "certification_status_must_match": True,
    "maximum_abs_certification_coverage_difference": 0.01,
    "maximum_abs_heldout_AURC_difference": 1.0e-4,
    "minimum_threshold_score_spearman": 0.999,
}


def archive_pre_repair() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    if (ARCHIVE / "archive_manifest.csv").exists():
        return
    sources: list[Path] = []
    for detector in clean.DETECTORS:
        for split in clean.SPLITS:
            for suffix in ("knn.json", "knn_cv.csv", "knn_bank.parquet"):
                sources.append(ART / "baselines" / f"{detector}_{split}_{suffix}")
            sources.append(ART / "policy_candidates" / f"{detector}_{split}_knn.json")
            sources.append(ART / "certification" / f"{detector}_{split}_knn.json")
            for partition in ("threshold_cal", "protocol_seen", "protocol_held_out"):
                sources.append(clean.score_path(detector, split, "knn", partition))
    sources.extend([
        ART / "policy_candidates" / "candidate_registry.csv",
        ART / "policy_candidates" / "candidate_freeze.json",
        ART / "certification" / "certification_registry.csv",
        ART / "certification" / "certification_trace.parquet",
        ART / "certification" / "policy_freeze.json",
    ])
    rows = []
    for path in sources:
        if not path.exists():
            continue
        rows.append({"relative_path": clean.rel(path), "size_bytes": path.stat().st_size, "sha256": clean.sha256_file(path)})
        # Preserve compact fitted/frozen artifacts. Large score files are
        # reproducible from the archived bank/model and retained by hash only.
        if "scores" not in path.parts:
            target = ARCHIVE / path.relative_to(ART)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    pd.DataFrame(rows).to_csv(ARCHIVE / "archive_manifest.csv", index=False)


def folded_risk_fit(detector: str, split: str) -> pd.DataFrame:
    features = pd.read_parquet(ART / "features" / detector / split / "risk_fit.parquet")
    folds = pd.read_parquet(ART / "oof_predictions" / f"{detector}_{split}_folds.parquet")
    out = features.merge(folds, on=["sample_id", "sha256"], validate="one_to_one")
    per_sha = out.groupby("sha256").cv_fold.nunique()
    if not per_sha.eq(1).all():
        raise RuntimeError(f"frozen clean-v2 fold assignment leaks SHA for {detector}/{split}")
    return out


def reference_rows(folded: pd.DataFrame, configuration: str) -> pd.DataFrame:
    ordered = folded.sort_values(["sha256", "sample_id"], kind="mergesort").reset_index(drop=True)
    if configuration == "all_rows":
        return ordered
    if configuration == FINAL_REFERENCE_CONFIGURATION:
        return ordered.drop_duplicates("sha256", keep="first").reset_index(drop=True)
    raise ValueError(configuration)


def select_k_grouped(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    detector: str,
    split: str,
    configuration: str,
    device: str,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    errors = frame.base_error.to_numpy(dtype=np.int64)
    sample_ids = frame.sample_id.astype(str).to_numpy()
    shas = frame.sha256.astype(str).to_numpy()
    folds = frame.cv_fold.to_numpy(dtype=np.int64)
    cv_rows, audit_rows = [], []
    max_k = max(CANDIDATE_K)
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = ~train
        train_sha, valid_sha = set(shas[train]), set(shas[valid])
        overlap = len(train_sha & valid_sha)
        audit_rows.append({
            "detector": detector, "split": split, "reference_configuration": configuration,
            "fold": int(fold), "train_rows": int(train.sum()), "validation_rows": int(valid.sum()),
            "train_unique_sha256": len(train_sha), "validation_unique_sha256": len(valid_sha),
            "cross_fold_SHA_groups": overlap, "status": "PASS" if overlap == 0 else "FAIL",
        })
        if overlap:
            raise RuntimeError(f"SHA leakage in grouped kNN CV: {detector}/{split}/fold={fold}")
        neighbors = exact_knn_neighbors(
            embeddings[train], embeddings[valid], max_k,
            bank_ids=sample_ids[train], query_ids=None, device=device, batch_size=1024,
        )
        cumulative = np.cumsum(neighbors["distances"], axis=1)
        for k in CANDIDATE_K:
            risk = cumulative[:, k - 1] / float(k)
            cv_rows.append({
                "detector": detector, "split": split, "reference_configuration": configuration,
                "fold": int(fold), "k": int(k),
                "error_detection_AUROC": float(roc_auc_score(errors[valid], risk)),
                "train_validation_sha_overlap": overlap,
            })
    cv = pd.DataFrame(cv_rows)
    means = (cv.groupby("k", as_index=False).error_detection_AUROC.mean()
             .rename(columns={"error_detection_AUROC": "mean_error_detection_AUROC"})
             .sort_values(["mean_error_detection_AUROC", "k"], ascending=[False, True], kind="mergesort"))
    selected = int(means.iloc[0].k)
    cv = cv.merge(means, on="k", validate="many_to_one")
    cv["selected_k"] = selected
    return selected, cv, pd.DataFrame(audit_rows)


def fit_grouped_configurations(device: str) -> pd.DataFrame:
    REPAIR.mkdir(parents=True, exist_ok=True)
    SENS.mkdir(parents=True, exist_ok=True)
    model_rows, all_cv, all_audit = [], [], []
    for detector in clean.DETECTORS:
        cache = Phase2Cache(ROOT, detector)
        for split in clean.SPLITS:
            folded = folded_risk_fit(detector, split)
            for configuration in ("all_rows", FINAL_REFERENCE_CONFIGURATION):
                frame = reference_rows(folded, configuration)
                embeddings = cache.embeddings_for(frame.sample_id)
                selected_k, cv, audit = select_k_grouped(
                    frame, embeddings, detector=detector, split=split,
                    configuration=configuration, device=device,
                )
                all_cv.append(cv); all_audit.append(audit)
                bank_path = (ART / "baselines" / f"{detector}_{split}_knn_bank.parquet"
                             if configuration == FINAL_REFERENCE_CONFIGURATION
                             else SENS / f"{detector}_{split}_all_rows_bank.parquet")
                bank = frame[["sample_id", "sha256", "label", "generator", "base_error", "cv_fold"]].copy()
                bank.to_parquet(bank_path, index=False)
                cv_path = (ART / "baselines" / f"{detector}_{split}_knn_cv.csv"
                           if configuration == FINAL_REFERENCE_CONFIGURATION
                           else SENS / f"{detector}_{split}_all_rows_cv.csv")
                cv.to_csv(cv_path, index=False)
                payload = {
                    "baseline": "cosine_knn", "model_version": "clean_v2_sha_grouped_knn_v2",
                    "detector": detector, "split": split, "source_partition": "risk_fit",
                    "reference_configuration": configuration, "reference_rows": len(bank),
                    "reference_unique_sha256": bank.sha256.nunique(), "selected_k": selected_k,
                    "candidate_k": list(CANDIDATE_K), "cross_validation_folds": 5,
                    "cv_assignment_source": clean.rel(ART / "oof_predictions" / f"{detector}_{split}_folds.parquet"),
                    "cv_assignment_sha256": clean.sha256_file(ART / "oof_predictions" / f"{detector}_{split}_folds.parquet"),
                    "sha_grouped_cv": True, "cross_fold_SHA_groups": int(audit.cross_fold_SHA_groups.sum()),
                    "selection_objective": "mean_detector_error_AUROC", "seed": clean.SEED,
                    "bank_sha256": clean.sha256_file(bank_path), "cv_sha256": clean.sha256_file(cv_path),
                    "risk_fit_feature_sha256": clean.sha256_file(ART / "features" / detector / split / "risk_fit.parquet"),
                    "q_registry_sha256": clean.sha256_file(ART / "base_thresholds.csv"),
                }
                payload["model_sha256"] = clean.payload_hash(payload)
                model_path = (ART / "baselines" / f"{detector}_{split}_knn.json"
                              if configuration == FINAL_REFERENCE_CONFIGURATION
                              else SENS / f"{detector}_{split}_all_rows_model.json")
                clean.write_json(model_path, payload)
                model_rows.append(payload)
                print(f"[{detector}/{split}/{configuration}] selected k={selected_k}", flush=True)
    cv_all = pd.concat(all_cv, ignore_index=True)
    audit_all = pd.concat(all_audit, ignore_index=True)
    cv_all.to_csv(REPAIR / "grouped_k_selection_cv.csv", index=False)
    audit_all.to_csv(REPAIR / "sha_grouped_cv_audit.csv", index=False)
    pd.DataFrame(model_rows).to_csv(REPAIR / "grouped_k_model_registry.csv", index=False)
    if not audit_all.cross_fold_SHA_groups.eq(0).all():
        raise RuntimeError("SHA_CV_LEAKAGE != 0")
    return pd.DataFrame(model_rows)


def model_and_bank(detector: str, split: str, configuration: str) -> tuple[dict[str, Any], pd.DataFrame]:
    if configuration == FINAL_REFERENCE_CONFIGURATION:
        model_path = ART / "baselines" / f"{detector}_{split}_knn.json"
        bank_path = ART / "baselines" / f"{detector}_{split}_knn_bank.parquet"
    else:
        model_path = SENS / f"{detector}_{split}_all_rows_model.json"
        bank_path = SENS / f"{detector}_{split}_all_rows_bank.parquet"
    return json.loads(model_path.read_text()), pd.read_parquet(bank_path)


def embedding_ids(cache: Phase2Cache, frame: pd.DataFrame) -> pd.Series:
    available = set(cache.index.sample_id.astype(str))
    if set(frame.sample_id.astype(str)).issubset(available):
        return frame.sample_id.astype(str).reset_index(drop=True)
    sha_map = (cache.predictions[["sha256", "sample_id"]]
               .sort_values(["sha256", "sample_id"], kind="mergesort")
               .drop_duplicates("sha256").rename(columns={"sample_id": "embedding_sample_id"}))
    resolved = frame[["sha256"]].merge(sha_map, on="sha256", how="left", validate="many_to_one")
    if resolved.embedding_sample_id.isna().any():
        raise RuntimeError("evaluation SHA missing from embedding cache")
    return resolved.embedding_sample_id.astype(str)


def score_partition(
    cache: Phase2Cache,
    detector: str,
    split: str,
    partition: str,
    frame: pd.DataFrame,
    configuration: str,
    device: str,
) -> pd.DataFrame:
    model, bank = model_and_bank(detector, split, configuration)
    bank_embeddings = cache.embeddings_for(bank.sample_id)
    query_ids = embedding_ids(cache, frame)
    query_embeddings = cache.embeddings_for(query_ids)
    result = clean.exact_knn_distance(
        bank_embeddings, query_embeddings, int(model["selected_k"]),
        bank_ids=bank.sample_id.astype(str).to_numpy(),
        query_ids=query_ids.to_numpy() if partition == "risk_fit" else None,
        device=device, batch_size=1024,
    )
    out = clean.score_frame_base(frame, detector, split, "knn", result["risk_score"])
    out["selected_k"] = int(model["selected_k"])
    out["fit_artifact_sha256"] = model["model_sha256"]
    out["reference_configuration"] = configuration
    return out


def score_threshold_cal(device: str) -> None:
    for detector in clean.DETECTORS:
        cache = Phase2Cache(ROOT, detector)
        for split in clean.SPLITS:
            frame = pd.read_parquet(ART / "features" / detector / split / "threshold_cal.parquet")
            final = score_partition(cache, detector, split, "threshold_cal", frame, FINAL_REFERENCE_CONFIGURATION, device)
            clean.save_score(final, detector, split, "knn", "threshold_cal")
            alternate = score_partition(cache, detector, split, "threshold_cal", frame, "all_rows", device)
            alt_path = SENS / "scores" / detector / split / "threshold_cal.parquet"
            alt_path.parent.mkdir(parents=True, exist_ok=True)
            alternate.to_parquet(alt_path, index=False)
            print(f"[{detector}/{split}] threshold_cal rescored", flush=True)


def candidate_records_from_scores(detector: str, split: str, scores: pd.DataFrame, *, id_prefix: str) -> list[dict[str, Any]]:
    assignment = pd.read_csv(clean.MAN / f"{detector}_{split}_policy_assignment.csv", usecols=["sha256", "calibration_subset"])
    merged = scores.merge(assignment, on="sha256", how="left", validate="many_to_one")
    select = merged[merged.calibration_subset.eq("policy_select")].drop(columns="calibration_subset")
    curve = clean.threshold_scan(select)
    feasible = curve[curve.select_feasible]
    if len(feasible):
        largest = int(feasible.accepted_count.max())
        targets = [(f, max(1, int(math.floor(largest * f))), "select_feasible_fraction")
                   for f in [1.00, .95, .90, .85, .80, .75, .70, .65, .60, .50]]
    else:
        total = int(curve.accepted_count.max())
        targets = [(f, max(1, int(math.floor(total * f))), "fallback_select_coverage")
                   for f in [.50, .40, .30, .20, .15, .10, .075, .05, .025, .01]]
    picked = []
    for fraction, target, source in targets:
        eligible = curve[(curve.accepted_count > 0) & (curve.accepted_count <= target)]
        if len(eligible):
            row = eligible.sort_values(["accepted_count", "threshold"], ascending=[False, False], kind="mergesort").iloc[0].to_dict()
            row.update({"target_fraction": fraction, "candidate_source": source})
            picked.append(row)
    dedup = {float(row["threshold"]): row for row in picked}
    ordered = sorted(dedup.values(), key=lambda x: (-float(x["threshold"]), -int(x["accepted_count"])))[:clean.K_MAX]
    records = []
    for rank, row in enumerate(ordered, 1):
        record = {
            "detector": detector, "split": split, "method": "knn", "alpha": clean.ALPHA,
            "policy": "source_group_cp", "candidate_id": f"{id_prefix}_{detector}_{split}_knn_C{rank:02d}",
            "candidate_rank": rank, "threshold": float(row["threshold"]),
            "select_accepted_count": int(row["accepted_count"]), "select_coverage": float(row["coverage"]),
            "select_error_count": int(row["accepted_errors"]), "select_empirical_risk": float(row["empirical_risk"]),
            "candidate_source": row["candidate_source"],
        }
        record["candidate_sha256"] = clean.payload_hash(record)
        records.append(record)
    return records


def freeze_final_candidates() -> pd.DataFrame:
    path = ART / "policy_candidates" / "candidate_registry.csv"
    registry = pd.read_csv(path)
    registry = registry[~registry.method.eq("knn")].copy()
    new_rows = []
    for detector in clean.DETECTORS:
        for split in clean.SPLITS:
            scores = pd.read_parquet(clean.score_path(detector, split, "knn", "threshold_cal"))
            rows = candidate_records_from_scores(detector, split, scores, id_prefix="repaired")
            new_rows.extend(rows)
            clean.write_json(ART / "policy_candidates" / f"{detector}_{split}_knn.json",
                             {"detector": detector, "split": split, "method": "knn", "candidate_count": len(rows),
                              "repair": "sha_grouped_cv_one_row_per_sha256_bank", "candidates": rows})
    out = pd.concat([registry, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(path, index=False)
    files = sorted((ART / "policy_candidates").glob("*.json"))
    files = [p for p in files if p.name != "candidate_freeze.json"] + [path]
    hashes = {clean.rel(p): clean.sha256_file(p) for p in files}
    clean.write_json(ART / "policy_candidates" / "candidate_freeze.json", {
        "status": "FROZEN_BEFORE_CERTIFICATION", "files": hashes, "freeze_sha256": clean.payload_hash(hashes),
        "policy_certify_scores_used": False, "policy_certify_labels_used_for_candidates": False,
        "knn_repair": "sha_grouped_cv_one_row_per_sha256_bank",
    })
    return out


def certify_records(detector: str, split: str, scores: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    assignment = pd.read_csv(clean.MAN / f"{detector}_{split}_policy_assignment.csv", usecols=["sha256", "calibration_subset"])
    certify = scores.merge(assignment, on="sha256", how="left", validate="many_to_one")
    certify = certify[certify.calibration_subset.eq("policy_certify")].drop(columns="calibration_subset")
    groups = clean.source_groups(certify)
    unique_groups = sorted(groups.unique())
    k_count, g_count = len(candidates), len(unique_groups)
    delta_cell = clean.DELTA / (k_count * g_count)
    trace, summaries = [], []
    for candidate in candidates.sort_values("candidate_rank", kind="mergesort").to_dict("records"):
        accepted_all = certify.risk_score.to_numpy(float) <= float(candidate["threshold"])
        bounds, counts, group_rows = {}, {}, []
        for group in unique_groups:
            mask = groups.eq(group).to_numpy(); accepted = accepted_all & mask
            n, k = int(accepted.sum()), int(certify.loc[accepted, "base_error"].sum())
            cp = clean.clopper_pearson_upper(k, n, delta_cell)
            passed = bool(n > 0 and cp <= clean.ALPHA)
            bounds[group] = cp
            counts[group] = {"group_size": int(mask.sum()), "accepted_count": n, "accepted_errors": k,
                             "empirical_selective_risk": k / n if n else None, "cp_upper": cp, "certified": passed}
            group_rows.append({"group": group, "certify_group_size": int(mask.sum()), "accepted_count": n,
                               "accepted_errors": k, "empirical_selective_risk": k / n if n else np.nan,
                               "cp_upper": cp, "group_certified": passed})
        candidate_passed = all(row["group_certified"] for row in group_rows)
        for row in group_rows:
            trace.append({"detector": detector, "split": split, "method": "knn", "alpha": clean.ALPHA,
                          "delta": clean.DELTA, "candidate_id": candidate["candidate_id"],
                          "candidate_rank": int(candidate["candidate_rank"]), "threshold": float(candidate["threshold"]),
                          "candidate_count_K": k_count, "group_count_G": g_count, "delta_cell": delta_cell,
                          **row, "candidate_certified": candidate_passed})
        summaries.append({"candidate_id": candidate["candidate_id"], "candidate_rank": int(candidate["candidate_rank"]),
                          "threshold": float(candidate["threshold"]), "certification_coverage": float(accepted_all.mean()),
                          "certification_accepted_count": int(accepted_all.sum()), "max_group_cp_upper": max(bounds.values()),
                          "candidate_certified": candidate_passed, "group_bounds": bounds, "group_counts": counts})
    passed = [row for row in summaries if row["candidate_certified"]]
    selected = sorted(passed, key=lambda x: (-x["threshold"], -x["certification_coverage"], x["max_group_cp_upper"], x["candidate_id"]))[0] if passed else None
    summary = {
        "detector": detector, "split": split,
        "certification_status": "CERTIFIED" if selected else "NO_CERTIFIED_THRESHOLD",
        "selected_threshold": selected["threshold"] if selected else None,
        "certification_coverage": selected["certification_coverage"] if selected else 0.0,
        "max_group_cp_upper": selected["max_group_cp_upper"] if selected else None,
        "candidate_count": k_count, "group_count": g_count, "delta_cell": delta_cell,
        "certification_counts": selected["group_counts"] if selected else {},
        "group_CP_bounds": selected["group_bounds"] if selected else {},
    }
    return pd.DataFrame(trace), summary


def certify_and_freeze() -> pd.DataFrame:
    # Recompute all method policies from their unchanged/final scores so every
    # policy points to the new global candidate freeze. Non-kNN numerical rows
    # are audited for equality after this call.
    before = pd.read_csv(ART / "certification" / "certification_registry.csv")
    before_nonknn = before[~before.method.eq("knn")].sort_values(["detector", "split", "method"]).reset_index(drop=True)
    result = clean.certify_policies(include_knn=True)
    after_nonknn = result[~result.method.eq("knn")].sort_values(["detector", "split", "method"]).reset_index(drop=True)
    compare_cols = ["detector", "split", "method", "certification_status", "selected_threshold",
                    "certification_coverage", "max_group_cp_upper", "candidate_count", "group_count", "delta_cell"]
    stable = True
    for column in compare_cols:
        left, right = before_nonknn[column], after_nonknn[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            stable = stable and bool(np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=0.0, atol=1e-15, equal_nan=True))
        else:
            stable = stable and left.astype("string").fillna("NA").equals(right.astype("string").fillna("NA"))
    clean.write_json(REPAIR / "non_knn_invariance_audit.json", {"unchanged": stable, "columns": compare_cols})
    if not stable:
        raise RuntimeError("non-kNN certification changed during scoped repair")
    return result


def score_evaluation(device: str) -> None:
    policy_freeze = clean.verify_policy_freeze()
    clean.write_json(REPAIR / "evaluation_opening_record.json", {
        "knn_policy_frozen_before_evaluation": True,
        "policy_freeze_sha256": policy_freeze["freeze_sha256"],
        "candidate_freeze_sha256": clean.verify_candidate_freeze()["freeze_sha256"],
        "final_reference_configuration": FINAL_REFERENCE_CONFIGURATION,
    })
    for detector in clean.DETECTORS:
        cache = Phase2Cache(ROOT, detector)
        for split in clean.SPLITS:
            for partition in ("protocol_seen", "protocol_held_out"):
                frame = pd.read_parquet(ART / "features" / detector / split / f"{partition}.parquet")
                final = score_partition(cache, detector, split, partition, frame, FINAL_REFERENCE_CONFIGURATION, device)
                clean.save_score(final, detector, split, "knn", partition)
                alternate = score_partition(cache, detector, split, partition, frame, "all_rows", device)
                path = SENS / "scores" / detector / split / f"{partition}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                alternate.to_parquet(path, index=False)
            print(f"[{detector}/{split}] evaluation rescored", flush=True)


def sensitivity_audit() -> pd.DataFrame:
    rows = []
    final_candidates = pd.read_csv(ART / "policy_candidates" / "candidate_registry.csv")
    final_cert = pd.read_csv(ART / "certification" / "certification_registry.csv")
    for detector in clean.DETECTORS:
        for split in clean.SPLITS:
            final_model, _ = model_and_bank(detector, split, FINAL_REFERENCE_CONFIGURATION)
            alt_model, _ = model_and_bank(detector, split, "all_rows")
            final_threshold = pd.read_parquet(clean.score_path(detector, split, "knn", "threshold_cal"))
            alt_threshold = pd.read_parquet(SENS / "scores" / detector / split / "threshold_cal.parquet")
            keys = ["sample_id", "sha256"]
            joined = final_threshold[keys + ["risk_score"]].rename(columns={"risk_score": "dedup_score"}).merge(
                alt_threshold[keys + ["risk_score"]].rename(columns={"risk_score": "all_rows_score"}),
                on=keys, validate="one_to_one")
            alt_records = candidate_records_from_scores(detector, split, alt_threshold, id_prefix="sensitivity_all_rows")
            alt_trace, alt_cert = certify_records(detector, split, alt_threshold, pd.DataFrame(alt_records))
            alt_trace.to_parquet(SENS / f"{detector}_{split}_all_rows_certification_trace.parquet", index=False)
            clean.write_json(SENS / f"{detector}_{split}_all_rows_certification.json", alt_cert)
            final_cert_row = final_cert[(final_cert.detector.eq(detector)) & final_cert.split.eq(split) & final_cert.method.eq("knn")].iloc[0]
            for partition in ("protocol_seen", "protocol_held_out"):
                final_eval = pd.read_parquet(clean.score_path(detector, split, "knn", partition))
                alt_eval = pd.read_parquet(SENS / "scores" / detector / split / f"{partition}.parquet")
                final_aurc = float(clean.calibrator_metrics(final_eval.base_error.to_numpy(int), final_eval.risk_score.to_numpy(float),
                                                            sample_ids=final_eval.sample_id.astype(str).to_numpy())["AURC"])
                alt_aurc = float(clean.calibrator_metrics(alt_eval.base_error.to_numpy(int), alt_eval.risk_score.to_numpy(float),
                                                          sample_ids=alt_eval.sample_id.astype(str).to_numpy())["AURC"])
                rows.append({
                    "detector": detector, "split": split, "partition": partition,
                    "dedup_selected_k": int(final_model["selected_k"]), "all_rows_selected_k": int(alt_model["selected_k"]),
                    "selected_k_equal": int(final_model["selected_k"]) == int(alt_model["selected_k"]),
                    "threshold_score_spearman": float(spearmanr(joined.dedup_score, joined.all_rows_score).statistic),
                    "threshold_score_mean_abs_difference": float(np.mean(np.abs(joined.dedup_score - joined.all_rows_score))),
                    "threshold_score_max_abs_difference": float(np.max(np.abs(joined.dedup_score - joined.all_rows_score))),
                    "dedup_certification_status": final_cert_row.certification_status,
                    "all_rows_certification_status": alt_cert["certification_status"],
                    "certification_status_equal": final_cert_row.certification_status == alt_cert["certification_status"],
                    "dedup_certification_coverage": float(final_cert_row.certification_coverage),
                    "all_rows_certification_coverage": float(alt_cert["certification_coverage"]),
                    "abs_certification_coverage_difference": abs(float(final_cert_row.certification_coverage) - float(alt_cert["certification_coverage"])),
                    "dedup_AURC": final_aurc, "all_rows_AURC": alt_aurc,
                    "abs_AURC_difference": abs(final_aurc - alt_aurc),
                })
    out = pd.DataFrame(rows)
    out["materiality_pass"] = (
        out.selected_k_equal
        & out.certification_status_equal
        & out.abs_certification_coverage_difference.le(MATERIALITY["maximum_abs_certification_coverage_difference"])
        & out.abs_AURC_difference.le(MATERIALITY["maximum_abs_heldout_AURC_difference"])
        & out.threshold_score_spearman.ge(MATERIALITY["minimum_threshold_score_spearman"])
    )
    out.to_csv(REPAIR / "reference_bank_sensitivity_audit.csv", index=False)
    clean.write_json(REPAIR / "reference_bank_sensitivity_materiality.json", {
        "criteria_frozen_before_run": MATERIALITY, "all_configurations_pass": bool(out.materiality_pass.all()),
    })
    return out


def update_metrics_and_reports() -> None:
    clean.evaluate_final(include_knn=True)
    clean.bootstrap_final()
    clean.certification_report()


def write_audit(model_registry: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    cv = pd.read_csv(REPAIR / "sha_grouped_cv_audit.csv")
    final_models = model_registry[model_registry.reference_configuration.eq(FINAL_REFERENCE_CONFIGURATION)]
    cert = pd.read_csv(ART / "certification" / "certification_registry.csv")
    knn_cert = cert[cert.method.eq("knn")]
    lines = [
        "# Clean-v2 cosine-kNN SHA-grouped CV Repair Audit", "",
        "The frozen clean-v2 SHA-grouped fold assignments were reused exactly. The final reference bank uses one canonical row per SHA-256.", "",
        "## Grouped-CV leakage", "", cv.to_markdown(index=False), "",
        "SHA_CV_LEAKAGE=0", "", "## Repaired selected k", "",
        final_models[["detector", "split", "selected_k", "reference_rows", "reference_unique_sha256", "model_sha256"]].to_markdown(index=False), "",
        "## Repaired certification", "", knn_cert.to_markdown(index=False), "",
        "## All-rows versus one-row/SHA sensitivity", "", sensitivity.to_markdown(index=False), "",
        f"REFERENCE_BANK_SENSITIVITY_PASS={str(bool(sensitivity.materiality_pass.all())).upper()}", "",
        "RiskGuard, margin-only, and MSP scores were not refitted. Their certification metrics were recomputed only to refresh global freeze provenance and were numerically invariant.",
    ]
    path = REP / "knn_sha_grouped_cv_repair_audit.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    clean.write_json(REPAIR / "final_decision.json", {
        "SHA_CV_LEAKAGE": 0,
        "reference_bank_configuration": FINAL_REFERENCE_CONFIGURATION,
        "reference_bank_sensitivity_pass": bool(sensitivity.materiality_pass.all()),
        "selected_k": {f"{r.detector}_{r.split}": int(r.selected_k) for r in final_models.itertuples()},
        "status": "PASS" if cv.cross_fold_SHA_groups.eq(0).all() and sensitivity.materiality_pass.all() else "FAIL",
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_pre_repair()
    model_registry = fit_grouped_configurations(args.device)
    score_threshold_cal(args.device)
    freeze_final_candidates()
    certify_and_freeze()
    score_evaluation(args.device)
    update_metrics_and_reports()
    sensitivity = sensitivity_audit()
    write_audit(model_registry, sensitivity)
    print("SHA_CV_LEAKAGE=0")
    print("REFERENCE_BANK_SENSITIVITY_PASS=", bool(sensitivity.materiality_pass.all()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
