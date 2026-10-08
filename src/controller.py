"""MRB-based workload constraints followed by validation-based depth selection.

This logic is NEW: neither notebook implements a controller. It consumes
probes.fit_mrb_region/prepare_mrb_records outputs and compiled workload
templates, without connecting, submitting jobs, or training a model.

Policy: fit N_2Q(MRB) = intercept + slope*d on per-depth mean COMPILED gate
counts. Map a QML circuit to d_equivalent = N_2Q(QML)/slope, then constrain
the relative polarization-loss proxy 1 - p**d_equivalent. The intercept is
reported but not subtracted from QML cost: MRB frame overhead is not free
QML work. A cancels from S(d)/S(0), so this budget excludes SPAM/frame loss.

Equal two-qubit counts do not make MRB and QML noise equivalent. Gate types,
directions, parallelism, one-qubit gates, durations, coherent errors, and drift
are not represented by this proxy. It is an explicit hackathon policy, not a
certified error probability or a prediction of ROC-AUC. Use fresh same-region
probe data, inspect fit diagnostics, and validate the chosen model on hardware.
"""

from numbers import Integral, Real

import numpy as np

from .execution import circuit_resources


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _probability(value, name, *, allow_one=True):
    if (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
            or value < 0 or value > 1 or (not allow_one and value == 1)):
        end = "1]" if allow_one else "1)"
        raise ValueError(f"{name} must be a finite number in [0, {end}.")
    return float(value)


def _region(qubits):
    qubits = [_integer(q, "physical qubit") for q in qubits]
    if len(qubits) != 4 or len(set(qubits)) != 4:
        raise ValueError("This controller requires four distinct physical qubits.")
    return qubits


def calibrate_mrb_cost(records, region):
    """Fit compiled two-qubit count per MRB depth for ONE physical region.

    records come from probes.prepare_mrb_records or process_mrb_results, with
    region, physical_qubits, depth, instance, and two_qubit_gates. Use records
    from one probe run, not pooled historical jobs. Each benchmark depth has
    equal fit weight, matching the polarization fit's per-depth means.
    Return JSON-compatible slope, intercept, R-squared, and observed means.
    """
    if not isinstance(region, str) or not region.strip():
        raise ValueError("Provide a nonempty physical-region name.")
    groups, identities, seen = {}, set(), set()
    for record in records:
        if record["region"] != region:
            continue
        depth = _integer(record["depth"], "MRB depth")
        if depth % 2:
            raise ValueError("The current MRB protocol uses even benchmark depths.")
        instance = _integer(record["instance"], "MRB instance")
        key = (depth, instance)
        if key in seen:
            raise ValueError("Duplicate depth/instance; supply a single MRB run.")
        seen.add(key)
        identities.add(tuple(_region(record["physical_qubits"])))
        groups.setdefault(depth, []).append(_integer(record["two_qubit_gates"], "gate count"))
    if len(groups) < 2 or len(identities) != 1:
        raise ValueError("Need at least two MRB depths on one consistent physical region.")
    depths = np.array(sorted(groups), dtype=float)
    costs = np.array([np.mean(groups[int(d)]) for d in depths])
    if np.ptp(costs) == 0:
        raise ValueError("MRB costs must grow with benchmark depth, not remain constant.")
    slope, intercept = np.linalg.lstsq(
        np.column_stack([depths, np.ones(len(depths))]), costs, rcond=None
    )[0]
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise ValueError("MRB two-qubit cost must have a positive fitted slope.")
    residual = float(np.sum((costs - (slope * depths + intercept)) ** 2))
    total = float(np.sum((costs - costs.mean()) ** 2))
    return {
        "region": region, "physical_qubits": list(next(iter(identities))),
        "cost_metric": "two_qubit_gates",
        "two_qubit_gates_per_mrb_depth": float(slope), "intercept": float(intercept),
        "r_squared": float(1.0 - residual / total),
        "depths": [int(d) for d in depths],
        "observed_costs": [{"depth": int(d), "instances": len(groups[int(d)]),
                            "mean_two_qubit_gates": float(cost)}
                           for d, cost in zip(depths, costs)],
    }


def model_resources(templates, backend):
    """Describe already compiled per-depth templates; never compile or run.

    Accept workload.prepare_final_templates (or legacy templates) and attach
    their physical layouts to execution.circuit_resources output. The controller
    consumes compiled cost rather than treating a model layer as an MRB layer.
    """
    if not templates:
        raise ValueError("Provide at least one compiled model template.")
    resources = {}
    for L, template in templates.items():
        L = _integer(L, "model depth", minimum=1)
        if template.get("L") != L:
            raise ValueError("Template depth does not match its mapping key.")
        resources[L] = {**circuit_resources(template["circuit"], backend),
                        "physical_qubits": _region(template["physical_qubits"])}
    return resources


def constrain_depths(mrb_fit, calibration, resources, *, max_polarization_loss,
                     p_lower=None, allow_extrapolation=False):
    """Return allowed depths and an auditable record of the hardware policy.

    mrb_fit is probes.fit_mrb_region's result from the SAME run/region used for
    calibration. resources is model_resources' depth->resource mapping.
    max_polarization_loss is caller-chosen, required, and lies in [0,1).
    Optionally supply a conservative lower p bound, e.g.
    np.quantile(probes.bootstrap_mrb_region(...)["p"], 0.025). This includes
    fit uncertainty only, not uncertainty of the cost mapping or hardware drift.

    Relative decay is bounded, not absolute S or measurement success. Default
    policy rejects equivalent depths above the largest measured MRB depth.
    The output is JSON-compatible; None for an unbounded mathematical limit
    means p=1, while the observed-range guard still applies. No unsafe fallback.
    """
    p = _probability(mrb_fit["p"], "fitted p")
    amplitude = mrb_fit["A"]
    if (isinstance(amplitude, bool) or not isinstance(amplitude, Real)
            or not np.isfinite(amplitude) or amplitude <= 0):
        raise ValueError("A nonpositive MRB amplitude cannot define relative polarization retention.")
    budget = _probability(max_polarization_loss, "max_polarization_loss", allow_one=False)
    if p_lower is not None:
        p_lower = _probability(p_lower, "p_lower")
        if p_lower > p:
            raise ValueError("A conservative p_lower cannot exceed fitted p.")
    p_used = p if p_lower is None else p_lower
    if not isinstance(allow_extrapolation, bool):
        raise ValueError("allow_extrapolation must be a boolean.")
    if calibration["cost_metric"] != "two_qubit_gates":
        raise ValueError("Use compiled two-qubit cost calibration.")
    slope = calibration["two_qubit_gates_per_mrb_depth"]
    if isinstance(slope, bool) or not isinstance(slope, Real) or not np.isfinite(slope) or slope <= 0:
        raise ValueError("Cost calibration slope must be finite and positive.")
    region_qubits = _region(calibration["physical_qubits"])
    fit_depths = np.asarray(mrb_fit["depths"], dtype=float)
    cost_depths = np.asarray(calibration["depths"], dtype=float)
    for depths in (fit_depths, cost_depths):
        if (depths.ndim != 1 or len(depths) < 2 or not np.isfinite(depths).all()
                or len(np.unique(depths)) != len(depths) or np.any(depths < 0)
                or np.any(depths % 2 != 0)):
            raise ValueError("Provide distinct nonnegative even measured MRB depths.")
    if set(fit_depths) != set(cost_depths):
        raise ValueError("Polarization fit and cost calibration must use the same MRB depths.")
    max_measured_depth = float(fit_depths.max())
    if not resources:
        raise ValueError("Provide compiled resources for at least one model depth.")
    if p_used == 1:
        depth_limit = None
    elif p_used == 0:
        depth_limit = 0.0
    else:
        depth_limit = float(np.log1p(-budget) / np.log(p_used))
    decisions = []
    for L in sorted(resources):
        L = _integer(L, "model depth", minimum=1)
        resource = resources[L]
        if _region(resource["physical_qubits"]) != region_qubits:
            raise ValueError("QML resources and MRB calibration must use the same ordered region.")
        gates = _integer(resource["two_qubit_gates"], "model two-qubit gate count")
        equivalent_depth = float(gates / slope)
        if equivalent_depth == 0 or p_used == 1:
            loss = 0.0
        elif p_used == 0:
            loss = 1.0
        else:
            loss = float(-np.expm1(equivalent_depth * np.log(p_used)))
        extrapolated = equivalent_depth > max_measured_depth + 1e-12
        within_budget = loss <= budget + 1e-12
        allowed = within_budget and (allow_extrapolation or not extrapolated)
        reason = ("beyond_measured_range" if extrapolated and not allow_extrapolation
                  else "over_budget" if not within_budget else "within_budget")
        decisions.append({
            "L": L, "two_qubit_gates": gates, "equivalent_mrb_depth": equivalent_depth,
            "predicted_polarization_loss": loss, "extrapolated": extrapolated,
            "allowed": allowed, "reason": reason,
        })
    return {
        "region": calibration["region"], "physical_qubits": region_qubits,
        "policy": "two_qubit_cost_relative_polarization_proxy",
        "max_polarization_loss": budget, "p_point_estimate": p, "p_used": p_used,
        "p_source": "point_estimate" if p_lower is None else "lower_bound",
        "equivalent_depth_limit": depth_limit, "max_measured_mrb_depth": max_measured_depth,
        "allow_extrapolation": allow_extrapolation, "calibration": calibration,
        "allowed_depths": [row["L"] for row in decisions if row["allowed"]],
        "depth_decisions": decisions,
    }


def select_depth(constraints, validation_summary, *, metric="roc_auc"):
    """Choose the highest validation metric among allowed depths; ties shallow.

    Accept evaluation.summarize_results' DataFrame or a sequence of row dicts.
    The metric must be a finite higher-is-better probability (default roc_auc;
    use metric='ROC_AUC' for the latest notebook's table). Training loss is not
    a selection metric. If region exists, only the constrained region is used;
    otherwise rows must be one explicitly supplied common/ideal validation set.
    All allowed depths need one metric each; compare on the same samples.
    No allowed depth returns selected_depth=None, rather than deploying L1.
    """
    allowed = [_integer(L, "allowed depth", minimum=1) for L in constraints["allowed_depths"]]
    if len(set(allowed)) != len(allowed):
        raise ValueError("Allowed depths must be distinct.")
    base = {"region": constraints["region"], "allowed_depths": allowed, "metric": metric}
    if not allowed:
        return {**base, "selected_depth": None, "validation_value": None,
                "reason": "no_allowed_depth"}
    if metric not in {"roc_auc", "ROC_AUC", "average_precision", "f1", "recall", "precision"}:
        raise ValueError("Choose a supported validation detection metric, not training loss.")
    rows = (validation_summary.to_dict("records") if hasattr(validation_summary, "to_dict")
            else list(validation_summary))
    scores = {}
    for row in rows:
        if "region" in row and row["region"] != constraints["region"]:
            continue
        L = _integer(row["L"], "validation depth", minimum=1)
        if L not in allowed:
            continue
        if L in scores:
            raise ValueError("Need one validation metric per allowed region/depth.")
        scores[L] = _probability(row[metric], f"validation {metric}")
    if set(scores) != set(allowed):
        raise ValueError("Provide validation metrics for every allowed depth in this region.")
    chosen = max(allowed, key=lambda L: (scores[L], -L))
    return {**base, "selected_depth": chosen, "validation_value": scores[chosen],
            "reason": "best_validation_among_allowed"}
