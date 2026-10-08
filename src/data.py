"""Local dataset preparation for the HTTP and UNSW anomaly experiments.

Cell references are zero-based. HTTP behavior preserves legacy C68-C73/C95.
UNSW follows teammate_latest.ipynb C126-C130: sparse one-hot encoding, scaling,
four-component TruncatedSVD, and angle scaling fitted on ALL normal training
rows before QML subsampling. No file is read or written on import.
"""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _binary_labels(y):
    labels = np.asarray(y)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be a nonempty one-dimensional array of 0/1 values.")
    return labels.astype(int)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def load_http(path):
    """Load the HDF5 MATLAB HTTP file into (X, y), preserving C68 orientation."""
    import h5py

    with h5py.File(path, "r") as handle:
        X = np.array(handle["X"]).T
        y = np.array(handle["y"]).reshape(-1)
    y = _binary_labels(y)
    if X.ndim != 2 or X.shape != (len(y), 3) or not np.isfinite(X).all():
        raise ValueError("HTTP data must contain three finite numeric features per row.")
    return X, y


def stratified_split(y, *, seed=42):
    """Return train/val/test row indices using the exact 70/15/15 split (C69)."""
    y = _binary_labels(y)
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=seed
    )
    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}


def select_normal_training(y, train_idx, *, size=2000, seed=42):
    """Sample normal training row indices without replacement (C70)."""
    y = _binary_labels(y)
    _positive_integer(size, "size")
    train_idx = np.asarray(train_idx)
    if (train_idx.ndim != 1 or not np.issubdtype(train_idx.dtype, np.integer)
            or np.any(train_idx < 0) or np.any(train_idx >= len(y))
            or len(np.unique(train_idx)) != len(train_idx)):
        raise ValueError("train_idx must contain distinct valid integer row indices.")
    normal = train_idx[y[train_idx] == 0]
    if len(normal) < size:
        raise ValueError(f"Requested {size} normal training rows; only {len(normal)} available.")
    return np.random.default_rng(seed).choice(normal, size=size, replace=False)


def to_quantum_angles(X_z):
    """Clip standardized features to +/-3 and map to +/-pi (C72)."""
    X_z = np.asarray(X_z)
    if X_z.ndim != 2 or not np.isfinite(X_z).all():
        raise ValueError("Features must be a finite two-dimensional numeric array.")
    return np.clip(X_z, -3.0, 3.0) * (np.pi / 3.0)


def _prepared_data(y, splits, qml_train_idx, angles, preprocessor, dataset, seed):
    return {
        **splits, "qml_train_idx": qml_train_idx,
        "X_train_q": angles["train"], "X_val_q": angles["val"], "X_test_q": angles["test"],
        "y_train": y[qml_train_idx], "y_val": y[splits["val_idx"]],
        "y_test": y[splits["test_idx"]], "preprocessor": preprocessor,
        "metadata": {
            "dataset": dataset, "seed": seed, "split": "stratified_70_15_15",
            "preprocessing_fit": "sampled_normal_training_only",
            "normal_training_rows": len(qml_train_idx),
            "quantum_features": angles["train"].shape[1], "angle_clip_z": 3.0,
            "sample_id_convention": "zero_based_row_position_in_loaded_file",
        },
    }


def prepare_http(X, y, *, n_train=2000, seed=42):
    """Split, sample normals, fit scaling, and prepare HTTP angles (C69-C73).

    Returns arrays, aligned labels, original row indices, the fitted scaler,
    and metadata. The scaler is fitted on the selected normals, not all train
    rows. Persist the returned indices/scaler to reproduce or resume a run.
    """
    X, y = np.asarray(X), _binary_labels(y)
    if X.shape != (len(y), 3) or not np.isfinite(X).all():
        raise ValueError("HTTP data must contain three finite numeric features per row.")
    splits = stratified_split(y, seed=seed)
    selected = select_normal_training(y, splits["train_idx"], size=n_train, seed=seed)
    scaler = StandardScaler().fit(X[selected])
    angles = {name: to_quantum_angles(scaler.transform(X[idx]))
              for name, idx in (("train", selected), ("val", splits["val_idx"]),
                                ("test", splits["test_idx"]))}
    return _prepared_data(y, splits, selected, angles, scaler, "http", seed)


def select_validation_samples(y_val, *, n_per_class=8, seed=123):
    """Return validation-relative indices, normals then anomalies (C95).

    For persistent sample IDs use prepared['val_idx'][returned_indices]. Reuse
    the same selection across all circuit depths and physical regions.
    Latest UNSW C138 uses n_per_class=32, seed=123; C171 uses 16, seed=777.
    """
    y_val = _binary_labels(y_val)
    _positive_integer(n_per_class, "n_per_class")
    rng = np.random.default_rng(seed)
    selections = []
    for label in (0, 1):
        candidates = np.where(y_val == label)[0]
        if len(candidates) < n_per_class:
            raise ValueError(f"Class {label} has fewer than {n_per_class} validation rows.")
        selections.append(rng.choice(candidates, size=n_per_class, replace=False))
    return np.concatenate(selections)


def load_unsw(path):
    """Load an UNSW-NB15 CSV, removing id and both label-bearing columns.

    Returns a feature DataFrame and binary y. Official CSV train/test filenames
    do not dictate this experiment's split: prepare_unsw partitions the supplied
    file internally. Record the input filename in the notebook's run config.
    """
    frame = pd.read_csv(path)
    if not frame.columns.is_unique or "label" not in frame:
        raise ValueError("UNSW CSV requires unique columns and a binary label column.")
    y = _binary_labels(frame["label"].to_numpy())
    X = frame.drop(columns=["id", "label", "attack_cat"], errors="ignore")
    if not len(X.columns):
        raise ValueError("UNSW CSV contains no input features.")
    return X, y


def prepare_unsw(X, y, *, n_train=2000, seed=42):
    """Reproduce latest UNSW preprocessing and QML sampling (C127-C130).

    Fit every preprocessing stage on ALL normal rows in the training partition.
    Sample n_train normal examples only AFTER compression/scaling/angle mapping.
    Numeric features are standardized, categorical features one-hot encoded,
    then TruncatedSVD compresses to four features. No imputation or PCA is used.
    The returned preprocessor is a fitted sklearn Pipeline for future inference;
    input columns and ordering must match the supplied feature DataFrame.
    """
    y = _binary_labels(y)
    if not isinstance(X, pd.DataFrame) or len(X) != len(y) or not X.columns.is_unique:
        raise ValueError("UNSW features must be a DataFrame aligned with labels.")
    if {"id", "label", "attack_cat"} & set(X.columns):
        raise ValueError("Remove id, label, and attack_cat before preprocessing.")
    # Include pandas' explicit string dtype as well as object, avoiding the
    # deprecated object-only selection while matching the notebook CSV columns.
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = [name for name in X.columns if name not in categorical]
    if not numeric:
        raise ValueError("UNSW preprocessing requires numeric features.")
    if X.isna().any().any() or not np.isfinite(X[numeric].to_numpy(dtype=float)).all():
        raise ValueError("UNSW features must not contain missing or infinite values.")

    splits = stratified_split(y, seed=seed)
    normal_idx = splits["train_idx"][y[splits["train_idx"]] == 0]
    _positive_integer(n_train, "n_train")
    if len(normal_idx) < max(n_train, 4):
        raise ValueError("Insufficient normal training rows for sampling/four-feature SVD.")
    features = ColumnTransformer(
        [("num", StandardScaler(), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical)],
        sparse_threshold=1.0,
    )
    Z_normal = features.fit_transform(X.iloc[normal_idx])
    Z_val = features.transform(X.iloc[splits["val_idx"]])
    Z_test = features.transform(X.iloc[splits["test_idx"]])
    if Z_normal.shape[1] < 4:
        raise ValueError("Four-feature SVD requires at least four encoded features.")
    compressor = TruncatedSVD(n_components=4, random_state=seed)
    normal_4 = compressor.fit_transform(Z_normal)
    val_4, test_4 = compressor.transform(Z_val), compressor.transform(Z_test)
    angle_scaler = StandardScaler()
    normal_q = to_quantum_angles(angle_scaler.fit_transform(normal_4))
    val_q = to_quantum_angles(angle_scaler.transform(val_4))
    test_q = to_quantum_angles(angle_scaler.transform(test_4))
    qml_choice = np.random.default_rng(seed).choice(len(normal_q), size=n_train, replace=False)
    selected = normal_idx[qml_choice]
    preprocessor = Pipeline([
        ("features", features), ("compressor", compressor), ("component_scaler", angle_scaler),
    ])
    angles = {"train": normal_q[qml_choice], "val": val_q, "test": test_q}
    prepared = _prepared_data(y, splits, selected, angles, preprocessor, "unsw_nb15", seed)
    prepared.update(normal_train_idx=normal_idx, qml_choice=qml_choice)
    prepared["metadata"].update(
        preprocessing_fit="all_normal_training_only", preprocessing_fit_rows=len(normal_idx),
        compression="truncated_svd_4", categorical_columns=categorical, numeric_columns=numeric,
        explained_variance_ratio=compressor.explained_variance_ratio_.tolist(),
        variance_captured=float(compressor.explained_variance_ratio_.sum()),
    )
    return prepared


def select_local_training(prepared, *, size=128, seed=2026):
    """Select the fixed local-training set from the QML pool (latest C146).

    Return pool-relative indices, original dataset sample IDs, angles, labels,
    and sampling settings. Does not alter the prepared dataset or fitted stages.
    """
    _positive_integer(size, "size")
    X, y = np.asarray(prepared["X_train_q"]), _binary_labels(prepared["y_train"])
    sample_ids = np.asarray(prepared["qml_train_idx"])
    if X.ndim != 2 or len(X) != len(y) or len(sample_ids) != len(y) or np.any(y != 0):
        raise ValueError("Local training requires an aligned normal-only QML pool.")
    if size > len(X):
        raise ValueError("Local training size exceeds the QML pool.")
    selected = np.random.default_rng(seed).choice(len(X), size=size, replace=False)
    return {"pool_indices": selected, "sample_ids": sample_ids[selected],
            "X_train_q": X[selected], "y_train": y[selected], "seed": seed, "size": size}
