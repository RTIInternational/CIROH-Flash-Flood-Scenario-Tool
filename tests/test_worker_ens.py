"""
test_worker_ens.py (v3) - end-to-end check of the new sac_worker.py, run the same way the
dashboard runs it (subprocess, JSON in / JSON out).  Run from code/ in the hydro-model env:

    python test_worker_ensemble.py 16010000
    python test_worker_ensemble.py 16010000 500      # member count (default 200)

Paste the whole output back.
"""
import sys, json, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

BASIN = sys.argv[1] if len(sys.argv) > 1 else "16010000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
WORKER = (Path(__file__).parent.parent / "src" / "sac_worker.py").resolve()
print("test_worker_ens v3 (tests sac_worker.py) | N =", N, "| basin", BASIN)

# verify worker file exists
if not WORKER.exists():
    sys.exit(f"STOP: {WORKER} does not exist.")
_src = WORKER.read_text(encoding="utf-8", errors="ignore")
if "WORKER_VERSION" not in _src:
    sys.exit(f"STOP: {WORKER} is missing WORKER_VERSION.")
print("worker file:", WORKER, "-> OK")

n = 24
times = [t.isoformat() for t in pd.date_range("2026-03-19T12:00:00", periods=n, freq="h")]
g = np.exp(-0.5 * ((np.arange(n) - 10) / 3.0) ** 2)
STORMS = {"moderate_60mm": 60.0, "large_120mm": 120.0, "extreme_250mm": 250.0}


def call(scen, ens=None):
    payload = {"basin": BASIN, "n_extend": 12, "event_start": times[0], "times": times,
               "scenarios": scen}
    if ens:
        payload["ensemble"] = ens
    t = time.time()
    r = subprocess.run([sys.executable, str(WORKER)], input=json.dumps(payload),
                       capture_output=True, text=True, timeout=900)
    dt = time.time() - t
    if r.returncode != 0:
        sys.exit(f"worker crashed:\n{r.stderr[-1500:]}")
    out = json.loads(r.stdout)
    if "error" in out:
        sys.exit(f"worker error: {out['error']}")
    return out, dt


scen = {k: (g / g.sum() * mm).tolist() for k, mm in STORMS.items()}
det, t_det = call(scen)
print(f"\nworker reports: {det.get('_worker_version')}")
print(f"[deterministic, no ensemble] {t_det:.1f}s (includes ~1-2 s spin-up); keys: "
      f"{sorted(k for k in det if not k.startswith('_'))}")
assert "_ensemble" not in det

cfg = {"n": N, "seed": 1, "sat": [0.4, 1.0], "ppt_mult": [0.9, 1.5],
       "mult": {"uztwm": [0.8, 1.2], "lzfsm": [0.7, 1.3], "lztwm": [0.8, 1.2], "rexp": [0.9, 1.1]},
       "uh_shift": [-1.0, 1.0]}
ens, t_ens = call(scen, cfg)
if "_ensemble_error" in ens:
    sys.exit(f"ENSEMBLE ERROR: {ens['_ensemble_error']}")
e = ens["_ensemble"]
print(f"\n[ensemble] {t_ens:.1f}s total for {len(scen)} scenarios x {N} members "
      f"({(t_ens - t_det) / (len(scen) * N) * 1000:.0f} ms/member) | {e['meta']['worker']} | {e['meta']['core']}")
print(f"spin-up saturation {e['meta']['sat_spinup']:.2f} (sampled range {cfg['sat']})")

same = all(ens[k] == det[k] for k in STORMS)
print(f"deterministic hydrographs unchanged by the ensemble run: {same}")

sat = np.array(e["members"]["saturation"])
print(f"\n{'storm':16s} {'det peak':>9s} {'p5':>8s} {'p50':>8s} {'p95':>8s} {'corr(peak,sat)':>15s} {'tpeak p5-p95 (h)':>18s}")
for k in STORMS:
    s = e["scenarios"][k]
    pk = np.array(s["peak"])
    tp = np.array(s["tpeak_h"])
    print(f"{k:16s} {max(det[k]):9.2f} {np.percentile(pk, 5):8.2f} {np.percentile(pk, 50):8.2f} "
          f"{np.percentile(pk, 95):8.2f} {np.corrcoef(pk, sat)[0, 1]:15.2f} {np.percentile(tp, 5):8.0f} - {np.percentile(tp, 95):.0f}")

# bands must be ordered; traces present; JSON size
for k in STORMS:
    p = [np.array(e["scenarios"][k]["pct"][str(q)]) for q in e["meta"]["percentiles"]]
    assert all((p[i] <= p[i + 1] + 1e-9).all() for i in range(len(p) - 1)), f"{k}: percentile bands cross"
print(f"\npercentile bands ordered: True | traces kept per scenario: {len(e['scenarios']['large_120mm']['traces'])} "
      f"| payload size {len(json.dumps(ens)) / 1e6:.1f} MB")

# a perturbation that should do nothing must reproduce the deterministic run
chk, _ = call({"large_120mm": scen["large_120mm"]}, {"n": 5, "seed": 1})
d = max(abs(np.array(chk["_ensemble"]["scenarios"]["large_120mm"]["pct"]["50"]) - np.array(det["large_120mm"])))
print(f"no-perturbation ensemble vs deterministic run, max difference: {d:.1e}")
print("\nDONE")
