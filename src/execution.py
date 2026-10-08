"""Shared circuit execution mechanics for MRB and QML.

Importing this module does not load credentials, connect, or submit jobs.
Measurement definitions, parameter-prefix choices, and scores belong to the
calling experiment. Circuit and result order is preserved throughout.
"""

import json
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path

from .backend import get_two_qubit_gate_names, validate_region


def compile_circuits(
    circuits, backend, physical_qubits, *, optimization_level, seed_transpiler=None
):
    """Compile a nonempty batch onto an ordered physical chain.

    Pass optimization_level=0 for legacy MRB or 1 for legacy QML. Measurements
    must already be present if wanted. Inputs may remain parameterized.
    Reject compiled operations outside the requested region: initial_layout
    alone does not guarantee that routing stays within it.
    """
    circuits = list(circuits)
    if not circuits:
        raise ValueError("Cannot compile an empty batch.")
    if (
        isinstance(optimization_level, bool)
        or not isinstance(optimization_level, Integral)
        or optimization_level not in range(4)
    ):
        raise ValueError("optimization_level must be an integer from 0 to 3.")
    region = validate_region(
        backend, physical_qubits, expected_size=circuits[0].num_qubits
    )
    if any(circuit.num_qubits != len(region) for circuit in circuits):
        raise ValueError("Every logical circuit must match the region size.")

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pm = generate_preset_pass_manager(
        backend=backend,
        initial_layout=region,
        optimization_level=int(optimization_level),
        seed_transpiler=seed_transpiler,
    )
    compiled = list(pm.run([circuit.copy() for circuit in circuits]))
    if len(compiled) != len(circuits):
        raise ValueError("Compiler returned an unexpected circuit count.")
    allowed = set(region)
    for circuit in compiled:
        active = {
            circuit.find_bit(qubit).index
            for instruction in circuit.data
            if instruction.operation.name != "barrier"
            for qubit in instruction.qubits
        }
        if not active <= allowed:
            raise ValueError("Compiled circuit uses qubits outside the requested region.")
    return compiled


def circuit_resources(circuit, backend) -> dict:
    """Return JSON-compatible compiled depth and gate counts (legacy C21/C26)."""
    operations = dict(circuit.count_ops())
    two_qubit_names = get_two_qubit_gate_names(backend)
    return {
        "transpiled_depth": circuit.depth(),
        "operations": operations,
        "two_qubit_gates": sum(
            count for name, count in operations.items() if name in two_qubit_names
        ),
    }


def bind_parameter_batch(circuit, assignments):
    """Return one fully bound copy per parameter mapping, in supplied order.

    Callers construct mappings using their feature and theta Parameter objects;
    this function does not infer parameter order or choose shared prefixes.
    """
    bound_circuits = []
    for mapping in assignments:
        bound = circuit.assign_parameters(mapping, inplace=False)
        if bound.num_parameters:
            raise ValueError("Parameter mapping leaves unbound circuit parameters.")
        bound_circuits.append(bound)
    if not bound_circuits:
        raise ValueError("Cannot bind an empty batch.")
    return bound_circuits


def create_sampler(backend):
    """Explicitly construct the legacy Runtime sampler; never called on import."""
    from qiskit_ibm_runtime import SamplerV2

    return SamplerV2(mode=backend)


def submit_circuits(sampler, circuits, *, shots):
    """Submit exactly one job and return immediately without waiting or retrying.

    This executes hardware when given a real Runtime sampler. Save the returned
    job ID and batch metadata before calling retrieve_results.
    """
    circuits = list(circuits)
    if not circuits:
        raise ValueError("Cannot submit an empty batch.")
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
        raise ValueError("shots must be a positive integer.")
    if any(circuit.num_parameters for circuit in circuits):
        raise ValueError("Bind all parameters before submission.")
    return sampler.run(circuits, shots=int(shots))


def retrieve_job(service, job_id):
    """Retrieve an existing job; this may make an authenticated request."""
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a nonempty string.")
    return service.job(job_id.strip())


def extract_counts(result, *, register_name, expected_count=None) -> list[dict]:
    """Decode one unbroadcast circuit result per entry, preserving order.

    Use register_name='meas' for MRB or 'readout' for QML. Return records with
    result_index, counts, and actual shots. Broadcast parameter PUBs are not
    supported because aggregating their counts would lose sample identity.
    """
    records = []
    for index, pub_result in enumerate(result):
        try:
            bits = getattr(pub_result.data, register_name)
        except AttributeError as exc:
            raise ValueError(f"Result {index} lacks register {register_name!r}.") from exc
        if bits.shape != ():
            raise ValueError("Expected scalar circuit results, not broadcast PUBs.")
        counts = dict(bits.get_counts())
        if any(not isinstance(n, Integral) or n < 0 for n in counts.values()):
            raise ValueError("Result contains invalid counts.")
        counts = {key: int(value) for key, value in counts.items()}
        total = sum(counts.values())
        if total <= 0:
            raise ValueError("Result contains no shots.")
        records.append({"result_index": index, "counts": counts, "shots": total})
    if expected_count is not None and len(records) != expected_count:
        raise ValueError("Result count does not match the submitted batch.")
    return records


def retrieve_results(job, *, register_name, expected_count=None) -> list[dict]:
    """Wait for a job and decode its results; may make authenticated requests."""
    return extract_counts(
        job.result(), register_name=register_name, expected_count=expected_count
    )


def save_submission_record(path, job_id, *, shots, circuit_metadata, config=None):
    """Save an ordered manifest before waiting; refuse to overwrite an old run.

    Supply plain JSON-compatible metadata, never credentials or circuit objects.
    The caller chooses the run directory and a distinct filename for each job.
    """
    record = {
        "job_id": job_id,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "shots": int(shots),
        "circuit_metadata": list(circuit_metadata),
        "config": {} if config is None else config,
    }
    serialized = json.dumps(record, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized + "\n")
