"""Four-qubit anomaly circuits, local training, and explicit hardware scoring.

Latest UNSW: mandatory encoding once, 11-parameter Ry/Rz/CRX identity-nested
blocks (teammate_latest C151), global score 1-P(0000) (C164), progressive
normal-only SPSA (C160/C166), and all-qubit hardware readout (C172/C174).
Cell references are zero-based. Older HTTP/shared-SPSA helpers are retained
below for reproducibility; their q3 score and 8-parameter blocks are distinct.
Importing this module never connects, loads credentials, or submits jobs.
"""

from numbers import Integral

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from . import execution


N_QUBITS = 4
N_FEATURES = 4
DEPTHS = (1, 2, 3, 4, 5)
PARAMS_PER_LAYER = 11


def _depth(L):
    if isinstance(L, bool) or not isinstance(L, Integral) or L not in DEPTHS:
        raise ValueError("L must be an integer from 1 to 5.")


def _features(X, width):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or X.shape[1] != width or not np.isfinite(X).all():
        raise ValueError(f"Provide a nonempty finite feature matrix with {width} columns.")
    return X


def _parameters(theta, required, *, exact=False):
    theta = np.asarray(theta, dtype=float)
    if (theta.ndim != 1 or not np.isfinite(theta).all()
            or (len(theta) != required if exact else len(theta) < required)):
        qualifier = "exactly" if exact else "at least"
        raise ValueError(f"Provide {qualifier} {required} finite theta values.")
    return theta


def build_unsw_nested_circuit(L):
    """Return (unmeasured circuit, x parameters, theta parameters), C151.

    Four fixed Ry(x[q]) encodings occur once. Every layer has eight Ry/Rz
    rotations, then CRX on 0->1, 1->2, 2->3. All eleven gates become identity
    at zero; appending eleven zeros preserves the shallower ideal state.
    Features and weights use separate ParameterVectors and binding maps.
    """
    _depth(L)
    x = ParameterVector("x", N_FEATURES)
    theta = ParameterVector("theta", PARAMS_PER_LAYER * L)
    qc = QuantumCircuit(N_QUBITS)
    for q in range(N_QUBITS):
        qc.ry(x[q], q)
    qc.barrier()
    for layer in range(L):
        b = PARAMS_PER_LAYER * layer
        for q in range(N_QUBITS):
            qc.ry(theta[b + 2 * q], q)
            qc.rz(theta[b + 2 * q + 1], q)
        qc.crx(theta[b + 8], 0, 1)
        qc.crx(theta[b + 9], 1, 2)
        qc.crx(theta[b + 10], 2, 3)
        qc.barrier()
    return qc, x, theta


def nested_scores(L, theta_values, X):
    """Exact local Statevector scores 1-P(0000), replacing C158 with C164.

    Return one score per input row. A longer shared vector is allowed and only
    its first 11L weights are bound. No shots, sampler, or backend are used.
    """
    qc, x, theta = build_unsw_nested_circuit(L)
    values = _parameters(theta_values, len(theta))
    X = _features(X, N_FEATURES)
    scores = []
    for sample in X:
        assignments = {param: float(value) for param, value in zip(x, sample)}
        assignments.update({param: float(values[i]) for i, param in enumerate(theta)})
        state = Statevector.from_instruction(qc.assign_parameters(assignments, inplace=False))
        # Roundoff can make P(0000) exceed one by ~1e-15. Keep valid scores
        # for evaluation.py without changing the objective or score direction.
        scores.append(float(np.clip(1.0 - float(np.abs(state.data[0]) ** 2), 0.0, 1.0)))
    return np.asarray(scores)


def nested_loss(L, theta_values, X):
    """Mean global score on normal training rows only (C164).

    Caller supplies normal-only data. A lower loss does not guarantee better
    attack detection; validation scores should be analysed in evaluation.py.
    """
    return float(nested_scores(L, theta_values, X).mean())


def train_nested_depth(L, theta_init, X_train, iterations=100, a=0.10, c=0.08,
                       seed=42, check_every=5, *, on_iteration=None):
    """Local exact SPSA of the newest eleven weights; freeze the prefix (C160).

    Return (best_theta, history), selecting by checked CENTER training losses,
    including iteration zero. Use the same entire fixed normal batch for all
    evaluations. Keep the notebook RNG seed+L and schedules 0.602/0.101.
    on_iteration(theta_copy, entry_copy) is called at recorded loss checks and
    allows caller-controlled reporting/checkpointing without file I/O here.
    """
    _depth(L)
    theta = _parameters(theta_init, PARAMS_PER_LAYER * L, exact=True).copy()
    X_train = _features(X_train, N_FEATURES)
    for name, value in (("iterations", iterations), ("check_every", check_every)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite([a, c]).all() or a <= 0 or c <= 0:
        raise ValueError("SPSA a/c must be finite and positive.")
    rng = np.random.default_rng(seed + L)
    train_slice = slice(PARAMS_PER_LAYER * (L - 1), PARAMS_PER_LAYER * L)
    best_theta = theta.copy()
    best_loss = nested_loss(L, theta, X_train)
    history = [{"iteration": 0, "loss": best_loss}]
    if on_iteration is not None:
        on_iteration(theta.copy(), history[0].copy())
    for k in range(iterations):
        a_k = a / ((k + 1) ** 0.602)
        c_k = c / ((k + 1) ** 0.101)
        delta = rng.choice([-1.0, 1.0], size=PARAMS_PER_LAYER)
        theta_plus, theta_minus = theta.copy(), theta.copy()
        theta_plus[train_slice] += c_k * delta
        theta_minus[train_slice] -= c_k * delta
        loss_plus = nested_loss(L, theta_plus, X_train)
        loss_minus = nested_loss(L, theta_minus, X_train)
        gradient = ((loss_plus - loss_minus) / (2 * c_k)) * delta
        theta[train_slice] -= a_k * gradient
        if (k + 1) % check_every == 0 or k == iterations - 1:
            center_loss = nested_loss(L, theta, X_train)
            entry = {"iteration": k + 1, "loss": center_loss}
            history.append(entry)
            if center_loss < best_loss:
                best_loss, best_theta = center_loss, theta.copy()
            if on_iteration is not None:
                on_iteration(theta.copy(), entry.copy())
    return best_theta, history


def train_progressive(X_train, *, depths=DEPTHS, iterations=100, a=0.10, c=0.08,
                      seed=42, check_every=5, on_depth=None):
    """Extract C166's progressive training loop, returning (models, histories).

    Depths must be consecutive from 1. L1 starts from N(0,0.05), seed 42 by
    default; deeper starts append eleven zeros to the preceding best model.
    Validation, plots, saving, and deciding which depth to deploy stay outside.
    on_depth(L, theta_copy, history_copy) runs after each trained depth.
    """
    depths = list(depths)
    if not depths:
        raise ValueError("Provide consecutive depths beginning at 1.")
    for L in depths:
        _depth(L)
    if depths != list(range(1, len(depths) + 1)):
        raise ValueError("Progressive depths must be consecutive beginning at 1.")
    X_train = _features(X_train, N_FEATURES)
    rng = np.random.default_rng(seed)
    models, histories = {}, {}
    prefix = None
    for L in depths:
        initial = (rng.normal(0.0, 0.05, PARAMS_PER_LAYER) if L == 1
                   else np.concatenate([prefix, np.zeros(PARAMS_PER_LAYER)]))
        theta, history = train_nested_depth(
            L, initial, X_train, iterations=iterations, a=a, c=c,
            seed=seed, check_every=check_every,
        )
        models[L], histories[L] = theta.copy(), history
        prefix = theta.copy()
        if on_depth is not None:
            on_depth(L, theta.copy(), [entry.copy() for entry in history])
    return models, histories


def prepare_final_templates(backend, physical_qubits, *, depths=DEPTHS,
                            seed_transpiler=None):
    """Compile latest all-four-qubit readout templates at level 1 (C172).

    Explicit backend/ordered region replace notebook globals. Compilation is
    delegated to execution.py; no sampler or job is created. Parameterized
    identity layers still cost hardware gates after compilation/binding.
    """
    depths = list(depths)
    if not depths or len(set(depths)) != len(depths):
        raise ValueError("Provide a nonempty sequence of distinct depths.")
    region = list(physical_qubits)
    templates = {}
    for L in depths:
        qc, x, theta = build_unsw_nested_circuit(L)
        meas = ClassicalRegister(N_QUBITS, "meas")
        qc.add_register(meas)
        for q in range(N_QUBITS):
            qc.measure(q, meas[q])
        compiled = execution.compile_circuits(
            [qc], backend, region, optimization_level=1, seed_transpiler=seed_transpiler
        )[0]
        templates[L] = {
            "circuit": compiled, "x_params": x, "theta_params": theta, "L": int(L),
            "encoding": "unsw_nested", "score": "global", "register_name": "meas",
            "physical_qubits": region.copy(),
        }
    return templates


def global_anomaly_scores(observations):
    """Convert all-qubit counts into 1-P(0000), preserving row order (C174)."""
    scores = []
    for observation in observations:
        counts = observation["counts"]
        if (not counts or any(not isinstance(key, str) or len(key) != N_QUBITS
                              or set(key) - {"0", "1"} for key in counts)
                or any(isinstance(value, bool) or not isinstance(value, Integral)
                       or value < 0 for value in counts.values())):
            raise ValueError("Expected four-bit strings and nonnegative integer counts.")
        total = sum(counts.values())
        if total <= 0:
            raise ValueError("Counts must contain at least one shot.")
        scores.append(1.0 - counts.get("0000", 0) / total)
    return np.asarray(scores)


def evaluate_final_region(region_name, templates, X_eval, models, *, sampler,
                          depths=None, shots=256, sample_ids=None, on_submitted=None):
    """Score current per-depth models in one explicit sampler job (C174).

    Return (scores, ordered metadata), depth-major then sample-major. models is
    a depth->theta mapping, e.g. train_progressive's result, replacing globals.
    Actual counts, shot totals, job ID, region, and stable sample IDs are retained.
    on_submitted(job, metadata_copy) runs before waiting so callers can persist
    a manifest. This executes hardware only if explicitly given a real sampler.
    """
    X_eval = _features(X_eval, N_FEATURES)
    depths = list(templates) if depths is None else list(depths)
    if not depths or len(set(depths)) != len(depths):
        raise ValueError("Provide a nonempty sequence of distinct depths.")
    if not isinstance(region_name, str) or not region_name.strip():
        raise ValueError("Provide a nonempty region name.")
    ids = None if sample_ids is None else list(sample_ids)
    if ids is not None and (len(ids) != len(X_eval) or len(set(ids)) != len(ids)):
        raise ValueError("sample_ids must be distinct and match the evaluation rows.")
    regions = set()
    for L in depths:
        _depth(L)
        if L not in templates or L not in models:
            raise ValueError("Every requested depth needs a template and a trained model.")
        template = templates[L]
        if (template.get("L") != L or template.get("encoding") != "unsw_nested"
                or template.get("score") != "global" or template.get("register_name") != "meas"
                or len(template["theta_params"]) != PARAMS_PER_LAYER * L
                or len(template["x_params"]) != N_FEATURES):
            raise ValueError("Use latest UNSW templates from prepare_final_templates.")
        _parameters(models[L], PARAMS_PER_LAYER * L, exact=True)
        regions.add(tuple(template["physical_qubits"]))
    if len(regions) != 1:
        raise ValueError("All templates must use the same physical region.")
    circuits, metadata = [], []
    for L in depths:
        template = templates[L]
        circuits.extend(bind_samples(template, models[L], X_eval))
        for i in range(len(X_eval)):
            entry = {"region": region_name, "L": int(L), "sample_index": i,
                     "encoding": "unsw_nested", "score": "global",
                     "physical_qubits": list(template["physical_qubits"])}
            if ids is not None:
                entry["sample_id"] = ids[i].item() if isinstance(ids[i], np.generic) else ids[i]
            metadata.append(entry)
    job = execution.submit_circuits(sampler, circuits, shots=shots)
    if on_submitted is not None:
        on_submitted(job, [{**meta, "physical_qubits": meta["physical_qubits"].copy()}
                           for meta in metadata])
    observations = execution.retrieve_results(job, register_name="meas", expected_count=len(circuits))
    scores = global_anomaly_scores(observations)
    return scores, [{**meta, **observation, "job_id": job.job_id()}
                    for meta, observation in zip(metadata, observations)]


def build_vad_circuit(L, *, encoding="http"):
    """Legacy HTTP/early reupload builder; not the latest UNSW architecture.

    Return (unmeasured circuit, feature parameters, theta parameters).

    HTTP: three features encoded once, including Ry/Rz/Ry on q3 (C74).
    Reupload: four features, Ry(x[q]) on each qubit before EVERY layer.
    Both modes use 8L trainable rotations and the same forward CNOT chain.
    """
    if isinstance(L, bool) or not isinstance(L, Integral) or L not in range(1, 6):
        raise ValueError("L must be an integer from 1 to 5.")
    if encoding not in ("http", "reupload"):
        raise ValueError("encoding must be 'http' or 'reupload'.")
    x = ParameterVector("x", 3 if encoding == "http" else 4)
    theta = ParameterVector("theta", 8 * L)
    qc = QuantumCircuit(N_QUBITS)
    if encoding == "http":
        for q in range(3):
            qc.ry(x[q], q)
        qc.ry(x[0], 3)
        qc.rz(x[1], 3)
        qc.ry(x[2], 3)
        qc.barrier()
    for layer in range(L):
        if encoding == "reupload":
            for q in range(N_QUBITS):
                qc.ry(x[q], q)
            qc.barrier()
        for q in range(N_QUBITS):
            offset = layer * 8 + q * 2
            qc.ry(theta[offset], q)
            qc.rz(theta[offset + 1], q)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.barrier()
    return qc, x, theta


def prepare_vad_template(L, backend, physical_qubits, *, encoding="http", seed_transpiler=None):
    """Build q3 readout and compile once at level 1 (C83).

    Return a template dictionary, including encoding and physical region so
    callers can record which workload was compiled. No sampler is created.
    """
    qc, x, theta = build_vad_circuit(L, encoding=encoding)
    qc.add_register(ClassicalRegister(1, "readout"))
    qc.measure(3, qc.cregs[0][0])
    region = list(physical_qubits)
    compiled = execution.compile_circuits(
        [qc], backend, region, optimization_level=1, seed_transpiler=seed_transpiler
    )[0]
    return {"circuit": compiled, "x_params": x, "theta_params": theta,
            "L": int(L), "encoding": encoding, "physical_qubits": region}


def prepare_templates(backend, physical_qubits, *, depths=(1, 2, 3, 4, 5),
                      encoding="http", seed_transpiler=None):
    """Legacy q3 templates (C87-C88); latest UNSW uses prepare_final_templates."""
    depths = list(depths)
    if not depths or len(set(depths)) != len(depths):
        raise ValueError("Provide a nonempty sequence of distinct depths.")
    region = list(physical_qubits)
    return {L: prepare_vad_template(L, backend, region, encoding=encoding,
                                    seed_transpiler=seed_transpiler) for L in depths}


def _validate_inputs(theta, X, templates, depths):
    depths = list(templates) if depths is None else list(depths)
    if not depths or len(set(depths)) != len(depths) or any(L not in templates for L in depths):
        raise ValueError("Requested depths must be distinct and present in templates.")
    feature_counts = {len(templates[L]["x_params"]) for L in depths}
    if len(feature_counts) != 1:
        raise ValueError("All templates must use the same feature width.")
    theta, X = np.asarray(theta, dtype=float), np.asarray(X, dtype=float)
    required = max(len(templates[L]["theta_params"]) for L in depths)
    if theta.ndim != 1 or len(theta) < required or not np.isfinite(theta).all():
        raise ValueError(f"Provide at least {required} finite shared theta values.")
    if X.ndim != 2 or not len(X) or X.shape[1] != next(iter(feature_counts)) or not np.isfinite(X).all():
        raise ValueError("Provide a nonempty finite feature matrix matching the templates.")
    for L in depths:
        if len(templates[L]["theta_params"]) != 8 * L:
            raise ValueError("Template parameter count does not match its depth.")
    identities = {(templates[L].get("encoding"), tuple(templates[L].get("physical_qubits", ())))
                  for L in depths}
    if len(identities) != 1:
        raise ValueError("Templates must share an encoding and physical region.")
    return theta, X, depths


def bind_samples(template, theta, X):
    """Bind feature samples and the appropriate theta prefix (C85/C90/C96)."""
    x, parameters = template["x_params"], template["theta_params"]
    theta, X = np.asarray(theta, dtype=float), np.asarray(X, dtype=float)
    if theta.ndim != 1 or len(theta) < len(parameters) or not np.isfinite(theta).all():
        raise ValueError("Shared theta vector is too short or invalid.")
    if X.ndim != 2 or not len(X) or X.shape[1] != len(x) or not np.isfinite(X).all():
        raise ValueError("Feature batch does not match the template.")
    mappings = [{**{param: float(value) for param, value in zip(x, sample)},
                 **{param: float(theta[i]) for i, param in enumerate(parameters)}} for sample in X]
    return execution.bind_parameter_batch(template["circuit"], mappings)


def anomaly_scores(observations):
    """Convert decoded single-bit readout counts to P(q3=1) (C85/C90/C96)."""
    scores = []
    for observation in observations:
        counts = observation["counts"]
        total = sum(counts.values())
        if not counts or set(counts) - {"0", "1"} or total <= 0:
            raise ValueError("Expected nonempty single-bit readout counts.")
        scores.append(counts.get("1", 0) / total)
    return np.array(scores)


def _score_circuits(sampler, circuits, metadata, shots, on_submitted):
    job = execution.submit_circuits(sampler, circuits, shots=shots)
    # A caller can persist the job ID/manifest before the blocking result call.
    if on_submitted is not None:
        on_submitted(job, metadata)
    observations = execution.retrieve_results(job, register_name="readout", expected_count=len(circuits))
    return anomaly_scores(observations), observations, job.job_id()


def score_batch(theta_values, X_batch, template, *, sampler, shots=256, on_submitted=None):
    """Score a legacy q3 template without recompilation (C85).

    Latest UNSW uses evaluate_final_region and its all-qubit global score.
    Reject mixed architectures before submitting a job.
    """
    if template.get("encoding") == "unsw_nested":
        raise ValueError("Use evaluate_final_region for latest UNSW global-score templates.")
    circuits = bind_samples(template, theta_values, X_batch)
    metadata = [{"L": template["L"], "sample_index": i} for i in range(len(circuits))]
    scores, _, _ = _score_circuits(sampler, circuits, metadata, shots, on_submitted)
    return scores


def evaluate_all_depths(theta, X_eval, templates, *, sampler, depths=None,
                        shots=256, sample_ids=None, on_submitted=None):
    """Score depth-major/sample-major in one job (C96).

    Return (scores, metadata). Metadata also retains counts, shots, job ID,
    encoding and physical qubits for persistence. sample_ids should identify
    original dataset rows; otherwise sample_index is the local batch position.
    """
    theta, X_eval, depths = _validate_inputs(theta, X_eval, templates, depths)
    ids = None if sample_ids is None else list(sample_ids)
    if ids is not None and (len(ids) != len(X_eval) or len(set(ids)) != len(ids)):
        raise ValueError("sample_ids must be distinct and match the evaluation rows.")
    circuits, metadata = [], []
    for L in depths:
        template = templates[L]
        circuits.extend(bind_samples(template, theta, X_eval))
        for i in range(len(X_eval)):
            entry = {"L": L, "sample_index": i, "encoding": template.get("encoding"),
                     "physical_qubits": template.get("physical_qubits")}
            if ids is not None:
                entry["sample_id"] = ids[i].item() if isinstance(ids[i], np.generic) else ids[i]
            metadata.append(entry)
    scores, observations, job_id = _score_circuits(sampler, circuits, metadata, shots, on_submitted)
    return scores, [{**meta, **observation, "job_id": job_id}
                    for meta, observation in zip(metadata, observations)]


def evaluate_shared_spsa_pair(theta_plus, theta_minus, X_batch, templates, *,
                              sampler, depths=None, shots=256, on_submitted=None):
    """Evaluate both perturbations/all depths in ONE job (C90).

    Ordering is perturbation -> depth -> sample. Objective is mean normal
    anomaly score, equally weighted across requested depths.
    """
    theta_plus, X_batch, depths = _validate_inputs(theta_plus, X_batch, templates, depths)
    theta_minus, _, _ = _validate_inputs(theta_minus, X_batch, templates, depths)
    if theta_plus.shape != theta_minus.shape:
        raise ValueError("Perturbation vectors must have the same shape.")
    circuits, metadata = [], []
    for perturbation, theta in (("plus", theta_plus), ("minus", theta_minus)):
        for L in depths:
            circuits.extend(bind_samples(templates[L], theta, X_batch))
            metadata.extend({"perturbation": perturbation, "L": L, "sample_index": i}
                            for i in range(len(X_batch)))
    scores, _, _ = _score_circuits(sampler, circuits, metadata, shots, on_submitted)
    block_size = len(depths) * len(X_batch)
    losses = []
    for block in (scores[:block_size], scores[block_size:]):
        losses.append({L: np.mean(block[j * len(X_batch):(j + 1) * len(X_batch)])
                       for j, L in enumerate(depths)})
    plus_losses, minus_losses = losses
    return np.mean(list(plus_losses.values())), np.mean(list(minus_losses.values())), plus_losses, minus_losses


def train_shared_spsa(theta_init, X_train, templates, *, sampler, depths=None,
                      iterations=5, batch_size=8, shots=256, a=0.15, c=0.10,
                      seed=42, on_submitted=None, on_iteration=None):
    """Legacy QPU shared-prefix SPSA, preserving schedules/RNG (C91).

    Latest UNSW trains locally using train_nested_depth/train_progressive.

    Caller must supply normal-only angles. Return (theta, history). Each loss
    is the average of +/- perturbation losses on a changing minibatch, NOT a
    post-update validation loss. on_iteration(theta_copy, history_entry) allows
    explicit checkpointing; on_submitted(job, metadata) runs before waiting.
    """
    theta, X_train, depths = _validate_inputs(theta_init, X_train, templates, depths)
    if len(theta) != max(len(templates[L]["theta_params"]) for L in depths):
        raise ValueError("Training theta length must match the deepest requested template.")
    for name, value in (("iterations", iterations), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if batch_size > len(X_train) or not np.isfinite([a, c]).all() or a <= 0 or c <= 0:
        raise ValueError("Batch size exceeds training rows, or SPSA a/c are invalid.")
    rng = np.random.default_rng(seed)
    theta = theta.copy()
    history = []
    for k in range(iterations):
        batch_idx = rng.choice(len(X_train), size=batch_size, replace=False)
        a_k = a / ((k + 1) ** 0.602)
        c_k = c / ((k + 1) ** 0.101)
        delta = rng.choice([-1.0, 1.0], size=len(theta))
        loss_plus, loss_minus, depth_plus, depth_minus = evaluate_shared_spsa_pair(
            theta + c_k * delta, theta - c_k * delta, X_train[batch_idx], templates,
            sampler=sampler, depths=depths, shots=shots, on_submitted=on_submitted,
        )
        gradient = ((loss_plus - loss_minus) / (2 * c_k)) * delta
        theta = theta - a_k * gradient
        entry = {"iteration": k + 1, "loss": (loss_plus + loss_minus) / 2,
                 "depth_losses": {L: (depth_plus[L] + depth_minus[L]) / 2 for L in depths}}
        history.append(entry)
        if on_iteration is not None:
            on_iteration(theta.copy(), {**entry, "batch_indices": batch_idx.tolist()})
    return theta, history
