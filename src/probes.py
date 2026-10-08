"""Four-qubit MRB generation and analysis extracted from legacynotebook.ipynb.

Legacy cell references below are zero-based. The v2 sampler, RNG ordering,
polarization formula, and fitting/bootstrap settings preserve the notebook.
Hardware submission and persistence are explicit calls to execution.py by the
notebook; importing or using the numerical helpers never submits a job.
"""

from numbers import Integral

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_clifford
from scipy.optimize import curve_fit

from .execution import compile_circuits, circuit_resources, extract_counts


N_QUBITS = 4


def make_pauli_layer(n_qubits, rng):
    """Independent I/X/Y/Z choices in logical-qubit order (C7)."""
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        p = rng.choice(["I", "X", "Y", "Z"])
        if p == "X":
            qc.x(q)
        elif p == "Y":
            qc.y(q)
        elif p == "Z":
            qc.z(q)
    return qc


def make_random_layer_v2(n_qubits, rng):
    """Uniform chain matching, with local Cliffords on idle qubits (C13)."""
    if n_qubits != N_QUBITS:
        raise ValueError("This sampler is currently designed for 4 qubits.")
    qc = QuantumCircuit(n_qubits)
    matchings = [[], [(0, 1)], [(1, 2)], [(2, 3)], [(0, 1), (2, 3)]]
    selected_edges = matchings[int(rng.integers(0, len(matchings)))]
    occupied_qubits = set()
    for control, target in selected_edges:
        qc.cx(control, target)
        occupied_qubits.update([control, target])
    for q in range(n_qubits):
        if q not in occupied_qubits:
            seed = int(rng.integers(0, 2**31 - 1))
            qc.append(random_clifford(1, seed=seed).to_instruction(), [q])
    return qc


def make_mrb_circuit_v2(depth, seed):
    """Build an unmeasured Pauli-dressed mirror circuit (C14).

    Depth counts forward plus mirrored randomized layers, not transpiled depth.
    Depth zero still includes the initial Clifford/Pauli frame and its inverse.
    """
    if isinstance(depth, bool) or not isinstance(depth, Integral) or depth < 0 or depth % 2:
        raise ValueError("MRB benchmark depth must be a nonnegative even integer.")
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(N_QUBITS)
    initial_frame = QuantumCircuit(N_QUBITS)
    for q in range(N_QUBITS):
        cliff_seed = int(rng.integers(0, 2**31 - 1))
        initial_frame.append(random_clifford(1, seed=cliff_seed).to_instruction(), [q])
    qc.compose(initial_frame, inplace=True)
    qc.barrier()
    qc.compose(make_pauli_layer(N_QUBITS, rng), inplace=True)
    qc.barrier()
    forward_layers = []
    for _ in range(depth // 2):
        layer = make_random_layer_v2(N_QUBITS, rng)
        forward_layers.append(layer)
        qc.compose(layer, inplace=True)
        qc.barrier()
        qc.compose(make_pauli_layer(N_QUBITS, rng), inplace=True)
        qc.barrier()
    for layer in reversed(forward_layers):
        qc.compose(layer.inverse(), inplace=True)
        qc.barrier()
        qc.compose(make_pauli_layer(N_QUBITS, rng), inplace=True)
        qc.barrier()
    qc.compose(initial_frame.inverse(), inplace=True)
    return qc


def ideal_target(circuit):
    """Return (most probable bitstring, probability) via Statevector (C16/C17)."""
    probs = Statevector.from_instruction(circuit).probabilities_dict()
    target, probability = max(probs.items(), key=lambda item: item[1])
    return target, float(probability)


def generate_mrb_records(depths, *, instances=8, base_seed=20261005):
    """Create depth-major records using base_seed + depth*100 + instance (C17).

    Settings remain caller-owned; refresh probes can use [0, 8, 32], instances=4.
    """
    depths = list(depths)
    if not depths or len(set(depths)) != len(depths):
        raise ValueError("Provide a nonempty sequence of distinct depths.")
    if isinstance(instances, bool) or not isinstance(instances, Integral) or not 1 <= instances <= 100:
        raise ValueError("instances must be between 1 and 100 to avoid legacy seed collisions.")
    records = []
    for depth in depths:
        for k in range(instances):
            seed = base_seed + depth * 100 + k
            circuit = make_mrb_circuit_v2(depth, seed)
            target, probability = ideal_target(circuit)
            if probability < 0.999999:
                raise ValueError(f"Ideal-target check failed at depth={depth}, instance={k}.")
            records.append({
                "depth": int(depth), "instance": k, "seed": int(seed),
                "target": target, "ideal_probability": probability, "circuit": circuit,
            })
    return records


def prepare_mrb_records(logical_records, backend, regions, *, seed_transpiler=None):
    """Measure and compile identical probes region-major at level 0 (C21/C26).

    Compilation/resources delegate to execution.py. No job is submitted.
    """
    logical_records = list(logical_records)
    if not logical_records or not regions:
        raise ValueError("Provide logical probe records and at least one region.")
    measured = []
    for record in logical_records:
        circuit = record["circuit"].copy()
        circuit.measure_all()
        measured.append(circuit)
    physical_records = []
    for region_name, physical_qubits in regions.items():
        region = list(physical_qubits)
        compiled = compile_circuits(
            measured, backend, region, optimization_level=0, seed_transpiler=seed_transpiler
        )
        for record, circuit in zip(logical_records, compiled):
            physical_records.append({
                **record, "region": region_name, "physical_qubits": region.copy(),
                **circuit_resources(circuit, backend), "circuit": circuit,
            })
    return physical_records


def build_mrb_manifest(physical_records):
    """Return JSON-compatible ordered metadata without circuit objects (C32)."""
    fields = ("region", "physical_qubits", "depth", "instance", "seed", "target",
              "ideal_probability", "transpiled_depth", "two_qubit_gates")
    return [
        {"result_index": index, **{key: record[key] for key in fields}}
        for index, record in enumerate(physical_records)
    ]


def hamming_distance(a, b):
    """Hamming distance after removing register spaces (C39)."""
    a, b = a.replace(" ", ""), b.replace(" ", "")
    if len(a) != len(b):
        raise ValueError(f"Bitstrings must have same length: {a}, {b}")
    return sum(bit_a != bit_b for bit_a, bit_b in zip(a, b))


def get_hamming_distribution(counts, target, n_qubits=4):
    """Return normalized distance bins h[0]...h[n_qubits] (C41)."""
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, Integral) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")
    target = target.replace(" ", "").zfill(n_qubits)
    if len(target) != n_qubits or set(target) - {"0", "1"}:
        raise ValueError("Target must be a binary bitstring of the specified width.")
    if any(isinstance(count, bool) or not isinstance(count, Integral) or count < 0 for count in counts.values()):
        raise ValueError("Counts must be nonnegative integers.")
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Counts must contain at least one shot.")
    h = [0.0] * (n_qubits + 1)
    for bitstring, count in counts.items():
        bitstring = bitstring.replace(" ", "").zfill(n_qubits)
        if len(bitstring) != n_qubits or set(bitstring) - {"0", "1"}:
            raise ValueError("Count keys must match the binary target width.")
        h[hamming_distance(bitstring, target)] += count / total
    return h


def effective_polarization(counts, target, n_qubits=4):
    """Return (S, h) using the unchanged legacy MRB formula (C44)."""
    h = get_hamming_distribution(counts, target, n_qubits)
    H = sum((-0.5)**k * h[k] for k in range(n_qubits + 1))
    S = (4**n_qubits) / (4**n_qubits - 1) * H - 1 / (4**n_qubits - 1)
    return S, h


def process_mrb_results(result, manifest):
    """Decode Runtime results and attach success/polarization (C37/C45).

    The manifest must retain exact submission order. Runtime decoding is shared
    through execution.py; returned records contain no circuit objects.
    """
    manifest = list(manifest)
    decoded = extract_counts(result, register_name="meas", expected_count=len(manifest))
    records = []
    for index, (meta, observation) in enumerate(zip(manifest, decoded)):
        if meta["result_index"] != index:
            raise ValueError("Manifest result_index does not match submission order.")
        target_count = observation["counts"].get(meta["target"], 0)
        S, h = effective_polarization(observation["counts"], meta["target"])
        records.append({
            **meta, **observation, "target_count": target_count,
            "success_probability": target_count / observation["shots"],
            "polarization": S, "hamming_distribution": h,
        })
    return records


def summarize_mrb(records):
    """Region/depth means and population stds, in first-seen order (C38/C47)."""
    groups = {}
    for record in records:
        groups.setdefault((record["region"], record["depth"]), []).append(record)
    return [
        {"region": region, "depth": depth, "instances": len(group),
         **{f"{metric}_{stat}": float(function([r[metric] for r in group]))
            for metric in ("success_probability", "polarization")
            for stat, function in (("mean", np.mean), ("std", np.std))}}
        for (region, depth), group in groups.items()
    ]


def decay_model(d, A, p):
    """Legacy exponential model (C48/C50); defined once here."""
    return A * (p ** d)


def _polarizations_by_depth(records, region, depths):
    depths = list(depths)
    if len(depths) < 2 or len(set(depths)) != len(depths):
        raise ValueError("Fitting requires at least two distinct depths.")
    data = {}
    for depth in depths:
        values = np.array([r["polarization"] for r in records
                           if r["region"] == region and r["depth"] == depth], dtype=float)
        if not len(values) or not np.all(np.isfinite(values)):
            raise ValueError(f"Missing or invalid polarization data: {region}, depth={depth}.")
        data[depth] = values
    return data


def fit_mrb_region(experiment_results, region, depths):
    """Fit A*p**d and four-qubit r_omega with legacy bounds/p0 (C48)."""
    data = _polarizations_by_depth(experiment_results, region, depths)
    depths = np.array(list(data), dtype=float)
    mean_S = np.array([np.mean(values) for values in data.values()])
    (A, p), _ = curve_fit(decay_model, depths, mean_S, p0=[mean_S[0], 0.99],
                         bounds=([0, 0], [1.5, 1]))
    return {"A": float(A), "p": float(p), "r_omega": float((255 / 256) * (1 - p)),
            "depths": depths, "mean_S": mean_S}


def bootstrap_mrb_region(experiment_results, region, depths, n_bootstrap=5000, seed=42):
    """Resample circuits within each depth, preserving legacy RNG/fits (C50).

    This estimates circuit-resampling uncertainty, not temporal hardware drift.
    Failed fits are skipped as in the notebook; all-failed runs raise an error.
    """
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    data = _polarizations_by_depth(experiment_results, region, depths)
    rng = np.random.default_rng(seed)
    samples = {"A": [], "p": [], "r_omega": []}
    for _ in range(n_bootstrap):
        means = np.array([np.mean(rng.choice(values, size=len(values), replace=True))
                          for values in data.values()])
        try:
            (A, p), _ = curve_fit(
                decay_model, np.array(list(data), dtype=float), means,
                p0=[means[0], 0.99], bounds=([0, 0], [1.5, 1]), maxfev=10000,
            )
        except (RuntimeError, ValueError):
            continue
        samples["A"].append(A)
        samples["p"].append(p)
        samples["r_omega"].append((255 / 256) * (1 - p))
    if not samples["p"]:
        raise ValueError("All bootstrap fits failed.")
    return {key: np.array(values) for key, values in samples.items()}
