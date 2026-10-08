"""IBM backend configuration; importing this module performs no I/O."""

from pathlib import Path
from numbers import Integral


REGIONS = {
    "0123": [0, 1, 2, 3],
    "6789": [6, 7, 8, 9],
}


def get_coupling_edges(backend) -> list[tuple[int, int]]:
    """Return directed hardware coupling edges (legacy notebook C3)."""
    if backend.coupling_map is None:
        raise ValueError("Backend does not expose an explicit coupling map.")
    return list(backend.coupling_map.get_edges())


def find_path(backend, length: int = 4) -> list[int] | None:
    """Find the first simple path using sorted, undirected DFS (legacy C4).

    This selects by connectivity, not calibration quality or gate direction.
    """
    if isinstance(length, bool) or not isinstance(length, Integral) or length < 1:
        raise ValueError("Path length must be a positive integer.")
    if length > backend.num_qubits:
        return None
    if length == 1:
        return [0] if backend.num_qubits else None

    adjacency = {}
    for a, b in get_coupling_edges(backend):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    def dfs(path):
        if len(path) == length:
            return path
        for neighbor in sorted(adjacency[path[-1]]):
            if neighbor not in path:
                result = dfs(path + [neighbor])
                if result is not None:
                    return result
        return None

    for start in sorted(adjacency):
        result = dfs([start])
        if result is not None:
            return result
    return None


def validate_region(backend, physical_qubits, *, expected_size: int = 4) -> list[int]:
    """Validate an ordered physical chain and return a fresh list.

    Connectivity is undirected, matching the legacy path search. Compilation
    must still handle native gate direction and verify the resulting layout.
    """
    qubits = list(physical_qubits)
    if len(qubits) != expected_size or not qubits:
        raise ValueError(f"Region must contain {expected_size} qubits.")
    if any(isinstance(q, bool) or not isinstance(q, Integral) for q in qubits):
        raise ValueError("Physical qubit indices must be integers.")
    qubits = [int(q) for q in qubits]
    if len(set(qubits)) != len(qubits):
        raise ValueError("Region must not contain duplicate qubits.")
    if any(q < 0 or q >= backend.num_qubits for q in qubits):
        raise ValueError("Region contains an out-of-range qubit index.")
    edges = set(get_coupling_edges(backend))
    for a, b in zip(qubits, qubits[1:]):
        if (a, b) not in edges and (b, a) not in edges:
            raise ValueError(f"Region is not an ordered chain: {a} and {b} are not coupled.")
    return qubits


def get_two_qubit_gate_names(backend) -> set[str]:
    """Identify native two-qubit operations from the target (legacy C25)."""
    return {
        name
        for name in backend.target.operation_names
        if getattr(backend.target.operation_from_name(name), "num_qubits", None) == 2
    }


def get_backend_metadata(backend, *, include_status: bool = False) -> dict:
    """Return serializable hardware metadata, never credentials.

    Set include_status=True explicitly to query operational/queue status;
    that call may make an authenticated request on a real IBM backend.
    Calibration extraction is not part of the current legacy implementation.
    """
    metadata = {
        "backend": backend.name,
        "num_qubits": int(backend.num_qubits),
        "coupling_edges": [list(edge) for edge in get_coupling_edges(backend)],
        "two_qubit_gate_names": sorted(get_two_qubit_gate_names(backend)),
    }
    if include_status:
        status = backend.status()
        metadata.update(
            operational=bool(status.operational),
            pending_jobs=int(status.pending_jobs),
            status_message=status.status_msg,
        )
    return metadata


def load_ibm_token() -> str:
    """Read the external token file only when explicitly called.

    No process-environment fallback or variable interpolation is used.
    The token must never be included in logs or experiment metadata.
    """
    from dotenv import dotenv_values

    env_path = Path.home() / ".qff26" / ".env"
    if not env_path.is_file():
        raise FileNotFoundError("IBM credentials file is missing: ~/.qff26/.env")

    values = dotenv_values(env_path, interpolate=False)
    token = values.get("IBM_QUANTUM_TOKEN")
    if not token or not token.strip():
        raise ValueError("Set IBM_QUANTUM_TOKEN in ~/.qff26/.env.")
    return token.strip()


def connect_backend(backend_name: str = "ibm_quebec", *, instance: str | None = None):
    """Connect explicitly and return (service, backend).

    This reads credentials and makes authenticated IBM requests. Call it only
    when ready to connect; it does not submit hardware jobs.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    options = {"channel": "ibm_cloud", "token": load_ibm_token()}
    if instance is not None:
        options["instance"] = instance
    service = QiskitRuntimeService(**options)
    return service, service.backend(backend_name)
