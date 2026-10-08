# Hardware-adaptive quantum anomaly detection

Team 03 — uOttawa Qiskit Fall Fest 2026.

This project combines a four-qubit cybersecurity anomaly detector with Mirror Randomized Benchmarking (MRB) on IBM's `ibm_quebec`. The detector learns from normal traffic only. MRB measures a physical region's circuit performance; a separate controller constrains allowable model depths using that measurement. Validation chooses the best detector among the allowed depths.

Healthier hardware can permit more depth. It does not imply that a deeper model detects attacks better.

## Repository

```text
notebooks/
  legacynotebook.ipynb      Original HTTP/MRB working reference
  teammate_latest.ipynb    Teammate's later experiments and handoff
  anomaly_experiment.ipynb UNSW preprocessing, local training, QPU evaluation
  hardware_probe.ipynb    MRB execution, saved analysis, controller demonstration
src/
  backend.py              Connection, physical regions, backend metadata
  execution.py            Compilation, layouts, Runtime batches, decoding, manifests
  probes.py               MRB circuits, targets, polarization, fits, bootstrap
  data.py                 HTTP/UNSW loading, splits, preprocessing, quantum angles
  workload.py             Detector circuits, local SPSA, scoring, QPU evaluation
  evaluation.py           Detection metrics and paired/sample bootstrap
  controller.py           Hardware constraints and validation-based selection
data/raw/                 Locally supplied datasets (ignored)
results/<run_name>/        Checkpoints, job manifests, counts, metrics, run figures
figures/                  Available for curated figures; notebooks save inside runs
tests/test_offline.py      Small offline checks of scientific invariants
```

The two reference notebooks are historical records, including superseded experiments and stateful cells. Use the orchestration notebooks for the current workflow. Region definitions and validation live in `backend.py`; no separate `regions.py` is needed.

## Setup

The recorded experiment used Python 3.13.13, Qiskit 2.5.2, Qiskit Runtime 0.50.0, and qiskit-machine-learning 0.9.1. Create a virtual environment from the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

In VS Code, select the `.venv` Python interpreter as the notebook kernel. Imports do not load credentials or contact IBM. The current detector uses explicit Qiskit circuits and manual SPSA rather than a qiskit-machine-learning estimator.

Supply `UNSW_NB15_training-set.csv` in `data/raw/`. It is not distributed with this repository. The recorded file has 175,341 rows and 45 columns. Its SHA-256 is stored in the run's `config.json`. The notebook creates its own stratified 70/15/15 split from this supplied file; it does not use the separately distributed UNSW test CSV. The older HTTP helpers accept `http.mat`.

For hardware access, create **outside the repository**:

```text
~/.qff26/.env
```

On Windows, `~` means the notebook Python's `Path.home()`, normally `C:\Users\<username>`. It does not mean the repository's parent folder. The file contains:

```dotenv
IBM_QUANTUM_TOKEN=your_private_token
```

`.env.example` contains only an empty placeholder. `backend.connect_backend()` explicitly loads the external file using python-dotenv, with no process-environment fallback. Hardware access requires an authorized IBM Cloud/PINQ² account with access to the requested backend. Never put a real token in a notebook, source file, run manifest, or Git.

## Current detector

`data.prepare_unsw()` removes `id`, `label`, and `attack_cat`; standardizes numeric columns and one-hot encodes categorical columns; compresses to four components with TruncatedSVD; and standardizes, clips to [-3, 3], and maps the components to angles via `z*pi/3`. **Every fitted preprocessing stage uses normal training rows only.** The saved run retained about 47.9% of the encoded-space variance.

The circuit encodes four inputs once using `Ry`, then applies L=1–5 trainable blocks. Each block has eight `Ry`/`Rz` rotations and three `CRX` entanglers along 0→1→2→3: **11 parameters per layer**, or 11/22/33/44/55 parameters. Appending eleven zero parameters leaves the ideal circuit state unchanged. This is the implemented CRX architecture, not earlier RZZ or repeated-encoding proposals.

The anomaly score is `1-P(0000)` measured on all four qubits. It is a circuit measurement probability, **not a calibrated probability that traffic is malicious**. Normal-only training minimizes the mean score. Progressive exact Statevector SPSA trains a new layer while freezing earlier weights. The starting layer is included when choosing the best checked training loss. Identity initialization preserves the earlier function; subsequent optimization can still harm validation performance.

The default local run uses 128 normal samples from a 2,000-sample normal pool, 100 iterations per depth, and a fixed balanced validation set of 32 normal plus 32 attack rows. Preprocessing and model weights are separate artifacts.

## Run the workflow

### 1. Local detector

Run `anomaly_experiment.ipynb` from the top with `CONNECT_TO_IBM=False` and `SUBMIT_QPU_JOBS=False`. Read its Markdown alongside the code. It writes a unique `results/unsw_nested_.../` directory with settings, preprocessing, sample IDs, models, local scores, metrics, and figures.

After training, the models are already frozen for evaluation: do not rerun the training cell. `models.npz` stores arrays `L1` through `L5`. To avoid retraining after a restart, set `MODEL_CHECKPOINT` to a compatible saved `models.npz` before running from the top. The checkpoint branch rebuilds preprocessing; it does **not** automatically restore another run's scaler/SVD. Keep the same CSV, software, split, and preprocessing configuration, and retain `preprocessor.joblib` and sample provenance. Load joblib artifacts only from a trusted source.

### 2. MRB

`hardware_probe.ipynb` defaults to saved analysis of the recorded MRB run, with connection/submission disabled. This needs no raw dataset or IBM access. It writes a new analysis directory while preserving the original timestamp and source.

To intentionally run a new probe, set `MODE="new"`, `CONNECT_TO_IBM=True`, and `SUBMIT_QPU_JOB=True`. Defaults are depths [0, 8, 32], four random instances each, and 256 shots. The same twelve logical circuits run on both regions [0,1,2,3] and [6,7,8,9]: **one job, 24 circuits, 6,144 shots**. Compilation uses optimization level 0 to retain the benchmark workload. Preserve `submission.json` immediately after submission.

To recover a submitted probe after interruption, use `MODE="recover"`, the original `RECOVERY_MANIFEST_PATH`, `CONNECT_TO_IBM=True`, and `SUBMIT_QPU_JOB=False`. Retrieve the original job rather than submitting again.

### 3. Frozen detector on hardware

In the anomaly notebook's existing local kernel, explicitly enable connection/submission immediately before section 14, then run sections 14–16. This does not train weights. Parameterized circuits compile at optimization level 1 on each selected region; data and frozen weights are then bound to those templates.

Defaults evaluate **16 normal plus 16 attack samples** at every depth: one job per region, 160 circuits and 40,960 shots per job. Both regions and the matching ideal reference use the same samples. Job IDs, ordered sample manifests, counts, and scores are persisted.

Do not run all cells to recover a job. Re-running the connection cell resets the anomaly notebook's in-memory submission guard. Use the saved job ID with `execution.retrieve_job()` and `execution.retrieve_results(register_name="meas", ...)`, then `workload.global_anomaly_scores()` and the saved manifest to rebuild results. Kernel guards cannot prevent duplicate jobs after a restart.

### 4. Hardware-adaptive decision

Before anomaly section 17, set `MRB_RESULTS_PATH` to one same-backend probe's `mrb_results.json` and choose `MAX_POLARIZATION_LOSS` explicitly. Run section 17 once hardware results exist. This is local analysis and submits no jobs. The recorded demonstration uses a **0.10 relative polarization-loss budget** and the fitted point estimate of p.

MRB fits `S(d)=A*p**d`. The controller fits mean compiled two-qubit gate cost against MRB depth, maps QML native gate counts to equivalent MRB depth, then constrains the proxy `1-p**equivalent_depth`. It rejects extrapolation beyond the measured depth range. Among allowed depths it selects the highest hardware validation ROC-AUC, breaking ties toward shallower depth. If none pass, selection is `None`.

The hardware notebook can demonstrate the same controller using `QML_RUN_DIR`, the anomaly run's `compiled_model_resources.json`, backend metadata, and `hardware_metrics.csv`. It optionally uses a bootstrap lower bound on p, so its decisions may differ from anomaly section 17's point-estimate policy. State which policy you report.

## Recorded results

The complete saved runs are:

- [QML run](results/unsw_nested_20261008T025411Z_a86f5fa1/): local training and two real-QPU evaluations.
- [MRB run](results/mrb_20261008T031414Z_5f6bd2/): one real-QPU refresh job.

Timestamps are UTC; these runs occurred on October 7 locally in Toronto. MRB's `recorded_at_utc` is its **submission timestamp**, not an exact execution-start time.

| L | Ideal AUC, local 64-row validation | Ideal AUC, matching 32-row hardware set | QPU AUC, 0123 | QPU AUC, 6789 |
|---|---:|---:|---:|---:|
| 1 | 0.9180 | 0.6367 | 0.6289 | 0.6367 |
| 2 | 0.9229 | 0.6367 | 0.6504 | 0.6563 |
| 3 | 0.9219 | 0.6445 | 0.6445 | 0.6504 |
| 4 | 0.9219 | 0.6445 | 0.6563 | 0.6660 |
| 5 | 0.9229 | 0.6445 | 0.6504 | 0.6641 |

Sources: `ideal_validation_report.csv` and `hardware_vs_matching_ideal.csv` in the QML run. **The 64-row and 32-row sets are different samples.** The weaker ideal result on the hardware set means the ~0.92-to-~0.64 difference cannot be attributed entirely to hardware noise. These small balanced sets do not establish production cybersecurity performance.

| Region | MRB fitted p | L1 loss proxy | Allowed depths at 10% budget | Selected depth |
|---|---:|---:|---|---|
| 0123 | 0.990605 | 5.34% | L1 | L1 |
| 6789 | 0.921341 | 37.91% | None | None |

Sources: the MRB fit and QML run's `controller_decisions.json`. L2 on 0123 narrowly fails at 10.40%. L1 is selected because it is the only allowable depth; this decision does not prove it has the highest unrestricted AUC. The chosen 10% policy was a demonstration choice, not an independently calibrated safety guarantee.

## Artifacts and Git

Keep run metadata, model weights, fitted preprocessing, job manifests, per-sample scores/counts, metrics, and figures together. `prepared_samples.npz` is a local dataset-derived cache and is ignored; reproduce its arrays from the supplied CSV and saved settings. The original recorded config captures initial hardware flags as false; job manifests and `run_status.json` record the later explicitly enabled hardware work. Historic absolute paths are provenance records, not paths to use on another computer.

Raw data, `.venv`, credentials, Python caches, and notebook checkpoints are ignored. `.env.example` is intentionally trackable with an empty token placeholder. Do not blanket-ignore `*.npz`: trained model checkpoints are useful submission artifacts. Notebooks retain executed results; publishing does not execute them, but a new Run All can train locally or, if switches are changed, submit new work.

## Offline checks

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

These checks use synthetic inputs and exact Statevector simulation only. They verify identity nesting, prefix freezing, MRB normalization, paired evaluation alignment, and controller behavior. They neither read the external token file nor connect to IBM. Saved real-QPU results can be reanalyzed without resubmitting jobs.

## Limits

- The controller is a gate-count/polarization heuristic, not a certified hardware error probability or a prediction of AUC. It excludes the fitted starting amplitude and ignores gate durations, one-qubit errors, gate-type differences, scheduling, and drift.
- The MRB refresh uses only three depths and four circuits per depth. Circuit-bootstrap intervals do not capture future hardware drift or all shot uncertainty. QML sample-bootstrap intervals do not capture hardware drift either.
- Normal-only loss does not explicitly push attacks upward. Lower training loss does not guarantee better detection.
- Four-component compression retains limited feature variance. The current architecture encodes once, with no per-layer data re-uploading; earlier gated/re-uploading experiments remain in the reference notebooks.
- Both validation sets are small and balanced. Precision and average precision depend on prevalence. The final held-out test remains disabled in the recorded run.
- MRB and QML jobs ran at different times. This is a completed integration demonstration, not evidence of continuous automatic adaptation or a live traffic deployment.
