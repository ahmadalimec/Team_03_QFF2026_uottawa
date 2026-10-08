"""Scientific invariants checked without credentials, network calls, or files."""

import unittest

import numpy as np
import pandas as pd

from src import controller, data, evaluation, probes, workload


class OfflineChecks(unittest.TestCase):
    def test_preprocessing_fits_normal_training_rows_only(self):
        rng = np.random.default_rng(42)
        labels = np.r_[np.zeros(60, dtype=int), np.ones(60, dtype=int)]
        X = pd.DataFrame(rng.normal(size=(120, 4)), columns=["a", "b", "c", "d"])
        X.loc[labels == 1, "a"] += 1000
        X["proto"] = np.where(labels == 0, "normal_protocol", "attack_protocol")
        prepared = data.prepare_unsw(X, labels, n_train=10)
        normal = prepared["normal_train_idx"]
        self.assertTrue(np.all(labels[normal] == 0))
        self.assertFalse(set(normal) & set(prepared["val_idx"]))
        self.assertFalse(set(normal) & set(prepared["test_idx"]))
        pipeline = prepared["preprocessor"]
        numeric_scaler = pipeline.named_steps["features"].named_transformers_["num"]
        np.testing.assert_allclose(numeric_scaler.mean_, X.iloc[normal][["a", "b", "c", "d"]].mean())
        encoder = pipeline.named_steps["features"].named_transformers_["cat"]
        self.assertEqual(encoder.categories_[0].tolist(), ["normal_protocol"])
        rebuilt = data.to_quantum_angles(pipeline.transform(X.iloc[prepared["val_idx"]]))
        np.testing.assert_allclose(rebuilt, prepared["X_val_q"])

    def test_zero_layer_preserves_every_depth(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-np.pi, np.pi, (3, 4))
        for L in range(1, 5):
            theta = rng.normal(size=11 * L)
            np.testing.assert_allclose(
                workload.nested_scores(L, theta, X),
                workload.nested_scores(L + 1, np.r_[theta, np.zeros(11)], X),
                atol=1e-12, rtol=0,
            )

    def test_progressive_training_preserves_prefix(self):
        X = np.array([[0.1, 0.2, 0.3, 0.4], [-0.2, 0.1, 0.4, 0.3]])
        models, _ = workload.train_progressive(X, depths=[1, 2], iterations=2, check_every=1)
        np.testing.assert_array_equal(models[1], models[2][:11])

    def test_mrb_known_target_and_normalization(self):
        for depth in [0, 8, 32]:
            target, probability = probes.ideal_target(probes.make_mrb_circuit_v2(depth, 42))
            self.assertGreater(probability, 0.999999)
            self.assertAlmostEqual(probes.effective_polarization({target: 256}, target)[0], 1)
            uniform = {format(i, "04b"): 16 for i in range(16)}
            self.assertAlmostEqual(probes.effective_polarization(uniform, target)[0], 0)

    def test_controller_uses_performance_and_refuses_unsafe_fallback(self):
        qs = [0, 1, 2, 3]
        records = [dict(region="0123", physical_qubits=qs, depth=d, instance=k,
                        two_qubit_gates=d) for d in [0, 8, 32] for k in range(2)]
        calibration = controller.calibrate_mrb_cost(records, "0123")
        fit = dict(A=0.95, p=0.99, depths=[0, 8, 32])
        resources = {L: dict(physical_qubits=qs, two_qubit_gates=6 * L) for L in range(1, 6)}
        constraints = controller.constrain_depths(
            fit, calibration, resources, max_polarization_loss=0.2,
        )
        metrics = [dict(region="0123", L=L, roc_auc=0.95 - L * 0.01) for L in range(1, 6)]
        self.assertGreater(len(constraints["allowed_depths"]), 1)
        self.assertEqual(controller.select_depth(constraints, metrics)["selected_depth"], 1)
        unsafe = controller.constrain_depths(fit, calibration, resources, max_polarization_loss=0.01)
        self.assertIsNone(controller.select_depth(unsafe, metrics)["selected_depth"])
        mismatched = {1: dict(physical_qubits=[6, 7, 8, 9], two_qubit_gates=6)}
        with self.assertRaises(ValueError):
            controller.constrain_depths(fit, calibration, mismatched, max_polarization_loss=0.2)

    def test_paired_auc_matches_samples_not_row_order(self):
        ref = pd.DataFrame(dict(region=["0123"] * 4, L=[1] * 4,
                                sample_id=[1, 2, 3, 4], true_label=[0, 0, 1, 1],
                                anomaly_score=[0.1, 0.2, 0.8, 0.9]))
        other = ref.assign(L=2).iloc[::-1]
        result = evaluation.paired_auc_difference(ref, other, n_bootstrap=20)
        self.assertEqual(result["auc_difference"], 0)
        np.testing.assert_array_equal(result["samples"], np.zeros(20))
        with self.assertRaises(ValueError):
            evaluation.paired_auc_difference(ref, other.iloc[:-1], n_bootstrap=20)


if __name__ == "__main__":
    unittest.main()
