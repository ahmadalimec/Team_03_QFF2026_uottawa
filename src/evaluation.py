"""Local anomaly-score analysis; no hardware or credential dependencies.

Zero-based legacy cells C98-C101 supply score tables, class means/separation,
and ROC-AUC. Average precision, threshold metrics, and stratified/paired
bootstrap intervals are new. Intervals describe sample-resampling uncertainty,
not QPU drift or shot uncertainty, and are not multiple-comparison adjusted.
"""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)


def _labels_and_scores(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if (y.ndim != 1 or scores.ndim != 1 or not len(y) or len(y) != len(scores)
            or not np.isin(y, [0, 1]).all()):
        raise ValueError("Provide aligned nonempty 1D binary labels and scores.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Anomaly scores must be finite probabilities in [0, 1].")
    return y.astype(int), scores


def results_to_frame(scores, metadata, y_eval, *, region=None):
    """Build the legacy C98 table while preserving workload metadata.

    Labels are indexed by sample_index (position within the evaluation batch).
    sample_id, when supplied by workload.py, identifies the original dataset
    row. Pass the named region explicitly for consistent cross-run grouping;
    otherwise it is derived from physical_qubits when available.
    """
    scores = np.asarray(scores, dtype=float)
    metadata = list(metadata)
    y_eval = np.asarray(y_eval)
    if (y_eval.ndim != 1 or not len(y_eval) or not np.isin(y_eval, [0, 1]).all()
            or scores.ndim != 1 or not len(scores) or len(scores) != len(metadata)):
        raise ValueError("Scores/metadata must align and y_eval must contain binary labels.")
    rows = []
    for score, meta in zip(scores, metadata):
        index = meta["sample_index"]
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(y_eval):
            raise ValueError("sample_index must identify a valid evaluation-batch row.")
        row = {**meta, "true_label": int(y_eval[index]), "anomaly_score": float(score)}
        if region is not None:
            row["region"] = region
        elif "region" not in row and row.get("physical_qubits") is not None:
            row["region"] = ",".join(str(q) for q in row["physical_qubits"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    _validate_frame(frame)
    return frame


def _validate_frame(frame):
    if not isinstance(frame, pd.DataFrame) or not {"true_label", "anomaly_score"} <= set(frame.columns):
        raise ValueError("Results must contain true_label and anomaly_score columns.")
    _labels_and_scores(frame["true_label"], frame["anomaly_score"])
    groups = [column for column in ("region", "L") if column in frame]
    if frame[groups].isna().any().any():
        raise ValueError("Result grouping fields must not be missing.")
    for key in ("sample_index", "sample_id"):
        if key in frame:
            if frame[key].isna().any() or frame.duplicated(groups + [key]).any():
                raise ValueError(f"Missing or duplicate {key} within a region/depth group.")


def detection_metrics(y, scores, *, threshold=None):
    """Class means/separation and ROC-AUC (C99/C101), plus new metrics.

    A threshold, if supplied, must be selected on validation data and frozen
    before test evaluation. Predict attack when score >= threshold. Precision
    and average precision reflect the prevalence of the supplied sample.
    """
    y, scores = _labels_and_scores(y, scores)
    if len(np.unique(y)) != 2:
        raise ValueError("ROC-AUC analysis requires both normal and anomaly samples.")
    normal_mean = float(np.mean(scores[y == 0]))
    anomaly_mean = float(np.mean(scores[y == 1]))
    metrics = {
        "n_samples": len(y), "n_anomalies": int(y.sum()), "prevalence": float(y.mean()),
        "normal_mean": normal_mean, "anomaly_mean": anomaly_mean,
        "separation": anomaly_mean - normal_mean,
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
    }
    if threshold is not None:
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be a finite probability in [0, 1].")
        predicted = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        metrics.update(
            threshold=float(threshold),
            precision=float(precision_score(y, predicted, zero_division=0)),
            recall=float(recall_score(y, predicted, zero_division=0)),
            f1=float(f1_score(y, predicted, zero_division=0)),
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        )
    return metrics


def summarize_results(frame, *, threshold=None):
    """Return one metrics row per region/depth, preserving first-seen order."""
    _validate_frame(frame)
    columns = [column for column in ("region", "L") if column in frame]
    if not columns:
        return pd.DataFrame([detection_metrics(frame.true_label, frame.anomaly_score, threshold=threshold)])
    rows = []
    for key, group in frame.groupby(columns, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(columns, key)),
                     **detection_metrics(group.true_label, group.anomaly_score, threshold=threshold)})
    return pd.DataFrame(rows)


def _bootstrap_settings(n_bootstrap, confidence):
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least 2.")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")


def _stratified_resamples(y, n_bootstrap, seed):
    normal, anomaly = np.where(y == 0)[0], np.where(y == 1)[0]
    if not len(normal) or not len(anomaly):
        raise ValueError("AUC bootstrap requires both classes.")
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        yield np.concatenate([rng.choice(normal, size=len(normal), replace=True),
                              rng.choice(anomaly, size=len(anomaly), replace=True)])


def _interval(samples, confidence):
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return {"ci_low": float(low), "ci_high": float(high),
            "bootstrap_std": float(np.std(samples, ddof=1)),
            "confidence": float(confidence), "n_bootstrap": len(samples)}


def bootstrap_auc(y, scores, *, n_bootstrap=5000, confidence=0.95, seed=42):
    """New stratified percentile interval, resampling within each class.

    Both classes are retained in every replicate. Return the point estimate,
    interval and raw samples; samples can be saved separately from JSON metrics.
    """
    y, scores = _labels_and_scores(y, scores)
    _bootstrap_settings(n_bootstrap, confidence)
    samples = np.array([roc_auc_score(y[idx], scores[idx])
                        for idx in _stratified_resamples(y, n_bootstrap, seed)])
    return {"roc_auc": float(roc_auc_score(y, scores)),
            **_interval(samples, confidence), "samples": samples}


def paired_auc_difference(reference, candidate, *, sample_key="sample_id",
                          n_bootstrap=5000, confidence=0.95, seed=42):
    """New paired percentile interval for AUC(reference) - AUC(candidate).

    Supply one region/depth group per frame. Align by stable sample ID, require
    identical sample sets/labels, then resample the SAME indices for both scores.
    Positive differences favor reference. sample_key='sample_index' is only
    appropriate when both frames refer to the exact same ordered input batch.
    """
    _bootstrap_settings(n_bootstrap, confidence)
    for frame in (reference, candidate):
        _validate_frame(frame)
        if sample_key not in frame or frame[sample_key].isna().any() or frame[sample_key].duplicated().any():
            raise ValueError("Paired comparison requires distinct nonmissing sample keys.")
        for column in ("region", "L"):
            if column in frame and frame[column].nunique() != 1:
                raise ValueError("Each paired input must contain a single region/depth group.")
    if set(reference[sample_key]) != set(candidate[sample_key]):
        raise ValueError("Paired comparison requires exactly the same samples.")
    # Canonical ordering also makes fixed-seed output invariant to input row order.
    aligned = reference.set_index(sample_key).sort_index()
    other = candidate.set_index(sample_key).loc[aligned.index]
    if not np.array_equal(aligned.true_label, other.true_label):
        raise ValueError("Paired sample labels disagree.")
    y, ref_scores = _labels_and_scores(aligned.true_label, aligned.anomaly_score)
    _, other_scores = _labels_and_scores(other.true_label, other.anomaly_score)
    samples = np.array([
        roc_auc_score(y[idx], ref_scores[idx]) - roc_auc_score(y[idx], other_scores[idx])
        for idx in _stratified_resamples(y, n_bootstrap, seed)
    ])
    ref_auc, other_auc = float(roc_auc_score(y, ref_scores)), float(roc_auc_score(y, other_scores))
    return {"reference_auc": ref_auc, "candidate_auc": other_auc,
            "auc_difference": ref_auc - other_auc,
            **_interval(samples, confidence), "samples": samples}


def compare_depths(frame, *, region, reference_depth=1, sample_key="sample_id",
                   n_bootstrap=5000, confidence=0.95, seed=42):
    """New paired comparisons of a reference depth against others in one region."""
    _validate_frame(frame)
    if not {"region", "L"} <= set(frame.columns):
        raise ValueError("Depth comparisons require region and L columns.")
    subset = frame[frame.region == region]
    reference = subset[subset.L == reference_depth]
    if reference.empty:
        raise ValueError("Reference region/depth has no results.")
    rows = []
    for depth in pd.unique(subset.L):
        if depth == reference_depth:
            continue
        result = paired_auc_difference(reference, subset[subset.L == depth], sample_key=sample_key,
                                       n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)
        rows.append({"region": region, "reference_depth": reference_depth, "candidate_depth": depth,
                     **{key: value for key, value in result.items() if key != "samples"}})
    return pd.DataFrame(rows)


def compare_regions(frame, *, depth, reference_region, sample_key="sample_id",
                    n_bootstrap=5000, confidence=0.95, seed=42):
    """New paired region comparisons at a fixed model depth."""
    _validate_frame(frame)
    if not {"region", "L"} <= set(frame.columns):
        raise ValueError("Region comparisons require region and L columns.")
    subset = frame[frame.L == depth]
    reference = subset[subset.region == reference_region]
    if reference.empty:
        raise ValueError("Reference region/depth has no results.")
    rows = []
    for region in pd.unique(subset.region):
        if region == reference_region:
            continue
        result = paired_auc_difference(reference, subset[subset.region == region], sample_key=sample_key,
                                       n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)
        rows.append({"L": depth, "reference_region": reference_region, "candidate_region": region,
                     **{key: value for key, value in result.items() if key != "samples"}})
    return pd.DataFrame(rows)
