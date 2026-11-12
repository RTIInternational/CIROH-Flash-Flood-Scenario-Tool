"""
test_inplace.py (v2) - which SAC-SMA parameters (and the UHG ordinates) can the ensemble
edit IN PLACE (reset + setattr + re-run) and still match a freshly built model?

Why: SAC-SMA may compute derived constants from some parameters when the object is
built.  If so, an in-place edit of such a parameter silently gives wrong results.
Each parameter is checked on its own, under spin-up states and under wet states.

Run from the code/ folder in the hydro-model env (needs the NEW sac_worker.py and ens_core.py):

    python test_inplace.py 16010000
    python test_inplace.py 16019000
    python test_inplace.py 16010000 --uhg-only     # skip the parameter table

Paste everything it prints back to Copilot.
"""
import sys
import json
import time
import tempfile
from pathlib import Path

# Add src/ to path so sac_worker and ens_core can be imported from root
_src_dir = (Path(__file__).parent.parent / "src").resolve()
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np
import pandas as pd
import pysacsnow as pss
import sac_worker as sw
import ens_core as ec

VERSION = "test_inplace_v2 (skips datetime arrays; has --uhg-only)"
print(VERSION, "|", ec.CORE_VERSION, "| pysacsnow", getattr(pss, "__version__", "unknown"))

BASIN = ([a for a in sys.argv[1:] if not a.startswith("--")] or ["16103000"])[0]
SAC_P, UHG_P, FORCING, UHG_CTRL = sw._paths_for_basin(BASIN)
BASE_SAC = json.loads(Path(SAC_P).read_text())
BASE_UHG = json.loads(Path(UHG_P).read_text())
BASE = {k: float(v[0]) for k, v in BASE_SAC.items() if k != "hru_id"}
UH0 = np.asarray(BASE_UHG["uh_ords"][0], float)

EVENT_START, N_EVENT, N_EXTEND = "2026-03-19T12:00:00", 24, 12
N_TOTAL = N_EVENT + N_EXTEND
RTOL = 1e-6
_h = np.arange(N_EVENT)
_g = np.exp(-0.5 * ((_h - 10) / 3.0) ** 2)
PPT = np.concatenate([_g / _g.sum() * 120.0, np.zeros(N_EXTEND)])
TIMES = pd.date_range(EVENT_START, periods=N_TOTAL, freq="h")
PET = (pd.read_csv(FORCING, parse_dates=["time"]).set_index("time")["hru_pet"]
       .reindex(TIMES, method="nearest").to_numpy(float))
SPIN, SPIN_DATE = sw._spinup(EVENT_START, SAC_P, FORCING)
WET = ec.states_from_saturation(np.array([0.9]), {k: np.array([v]) for k, v in BASE.items()}, SPIN)
WET = {k: float(v[0]) for k, v in WET.items()}
print(f"basin {BASIN} | spin-up state date {SPIN_DATE} | spin-up saturation "
      f"{ec.saturation_of(SPIN, BASE):.2f}")


def _tmp(suffix):
    return Path(tempfile.mktemp(suffix=suffix))


def write_sac(**override):
    out = {k: (list(v) if k == "hru_id" else [float(v[0])]) for k, v in BASE_SAC.items()}
    for k, val in override.items():
        out[k] = [float(val)]
    p = _tmp(".json")
    p.write_text(json.dumps(out))
    return p


def write_uhg(ords):
    p = _tmp(".json")
    p.write_text(json.dumps({"hru_id": BASE_UHG["hru_id"], "hru_area": BASE_UHG["hru_area"],
                             "uh_ords": [[float(x) for x in ords]]}))
    return p


def build(sac_params, uhg_params):
    """Same construction as sac_worker.main()."""
    sc = pss.Controller.load_from_yaml(sw._SAC_CTRL_TEMPLATE, verbose=False)
    sc.config.model.parameters.path = sac_params
    sc.config.model.forcing[0].path = FORCING
    sc.config.simulation.output_path = _tmp(".csv")
    uc = pss.Controller.load_from_yaml(UHG_CTRL, verbose=False)
    uc.config.model.parameters.path = uhg_params
    uc.config.model.forcing[0].path = FORCING
    uc.config.simulation.output_path = _tmp(".csv")
    return sc, uc, pss.SACSMA(control=sc, calc_method="numpy"), pss.UHG(control=uc, calc_method="numpy")


def run_event(m, states):
    """Same stepping loop as sac_worker._run_scenario."""
    sc, uc, sac, uhg = m
    sac.reset(); uhg.reset()
    for var, val in states.items():
        getattr(sac, var)[:] = np.array([val])
    for i in range(N_TOTAL):
        sc.advance(); sac.advance()
        sac.hru_ppt[:] = PPT[i]; sac.hru_pet[:] = PET[i]; sac.hru_aesc[:] = 0.0
        sac.calculate(); sac.output()
        uc.advance(); uhg.advance()
        uhg.hru_tci[:] = sac.tci[:]
        uhg.calculate(); uhg.output()
    return np.sum(uhg.buffer_states["q"][:N_TOTAL], axis=1)


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def new_value(k):
    v = BASE[k]
    if k in ("pctim", "adimp") and v < 0.02:
        return v + 0.03                       # multiplicative change would do nothing at 0
    if k in ("lzpk", "lzsk", "uzk"):
        return v * 1.15                       # keep ordering/bounds sane
    return v * 1.2


UHG_ONLY = "--uhg-only" in sys.argv
CANDIDATES = [] if UHG_ONLY else ["uztwm", "uzfwm", "lztwm", "lzfpm", "lzfsm", "adimp", "uzk", "lzpk", "lzsk",
              "zperc", "rexp", "pctim", "pfree"]
fixed = {"dry": SPIN, "wet": WET}
m_base = build(SAC_P, UHG_P)
REF = {name: run_event(m_base, st) for name, st in fixed.items()}   # calibrated, fresh model

RESULT = {}
print(f"\n{'param':8s} {'base':>10s} {'test':>10s}  {'in-place vs fresh (dry/wet)':>30s}  {'effect (dry/wet)':>18s}  verdict")
for k in CANDIDATES:
    if not hasattr(m_base[2], k):
        RESULT[k] = "MISSING"
        print(f"{k:8s} {BASE.get(k, float('nan')):10.4g} {'':>10s}  attribute not found on SACSMA -> cannot edit in place")
        continue
    v = new_value(k)
    errs, effs = [], []
    try:
        m_fresh = build(write_sac(**{k: v}), UHG_P)
        for name, st in fixed.items():
            q_fresh = run_event(m_fresh, st)
            getattr(m_base[2], k)[:] = v                        # in-place edit on the long-lived model
            q_inpl = run_event(m_base, st)
            getattr(m_base[2], k)[:] = BASE[k]                  # restore
            errs.append(rel(q_inpl, q_fresh))
            effs.append(rel(q_fresh, REF[name]))
        back = max(rel(run_event(m_base, st), REF[n]) for n, st in fixed.items())
    except Exception as exc:
        RESULT[k] = f"ERROR {type(exc).__name__}"
        print(f"{k:8s} error: {type(exc).__name__}: {exc}")
        continue
    if max(errs) >= RTOL or back >= RTOL:
        verdict = "FAIL (stale derived value or no restore)"
    elif max(effs) < 1e-9:
        verdict = "INCONCLUSIVE (no effect in this storm)"
    else:
        verdict = "PASS"
    RESULT[k] = verdict.split()[0]
    print(f"{k:8s} {BASE[k]:10.4g} {v:10.4g}  {errs[0]:14.1e} /{errs[1]:.1e}  {effs[0]:8.1%} /{effs[1]:.1%}  {verdict}")

# all passing parameters edited together
good = [k for k, r in RESULT.items() if r == "PASS"]
if len(good) > 1:
    ov = {k: new_value(k) for k in good}
    m_f = build(write_sac(**ov), UHG_P)
    q_f = run_event(m_f, WET)
    for k, v in ov.items():
        getattr(m_base[2], k)[:] = v
    q_i = run_event(m_base, WET)
    print(f"\nall {len(good)} PASS parameters edited together: in-place vs fresh {rel(q_i, q_f):.1e}  "
          f"{'PASS' if rel(q_i, q_f) < RTOL else 'FAIL -> interaction problem'}")
    for k in ov:
        getattr(m_base[2], k)[:] = BASE[k]

# unit hydrograph ordinates
print("\nUHG ordinates (volume-conserving 1 h shift):")
uhg_obj = m_base[3]
hits, arrays = [], []
for a in dir(uhg_obj):
    if a.startswith("__"):
        continue
    try:
        v = getattr(uhg_obj, a)
    except Exception:
        continue
    if isinstance(v, np.ndarray):
        arrays.append(f"{a}{v.shape}:{v.dtype}")
        if v.dtype.kind != "f":              # skip datetime / int / object arrays
            continue
        flat = v.reshape(-1)
        if flat.size >= UH0.size and np.allclose(flat[:UH0.size], UH0):
            hits.append(a)
UHG_ATTR = None
if not hits:
    print("  no ndarray attribute holds the ordinates. UHG arrays found:", ", ".join(arrays) or "none")
else:
    new = ec.scale_uh(UH0, 1.0)
    q_fresh = run_event(build(SAC_P, write_uhg(new)), SPIN)
    q_ref = run_event(build(SAC_P, UHG_P), SPIN)
    for a in hits:
        tgt = getattr(m_base[3], a)
        saved = tgt.copy()
        flat = tgt.reshape(-1)
        flat[:UH0.size] = new
        q_inpl = run_event(m_base, SPIN)
        tgt[...] = saved
        e, moved = rel(q_inpl, q_fresh), rel(q_fresh, q_ref)
        ok = e < RTOL and moved > 1e-9
        print(f"  attribute '{a}' shape {saved.shape}: in-place vs fresh {e:.1e}, shift changed output {moved:.1%} -> "
              f"{'PASS' if ok else 'FAIL'}")
        if ok and UHG_ATTR is None:
            UHG_ATTR = a

print("\n================ paste these lines into sac_worker.py ================")
safe = sorted({k for k, r in RESULT.items() if r == "PASS"} | ({"uzk", "zperc"} - {k for k, r in RESULT.items() if r in ("FAIL", "MISSING") or r.startswith("ERROR")}))
print("(parameter table skipped)" if UHG_ONLY else f"_INPLACE_SAFE = frozenset({set(safe)})")
print(f"_UHG_INPLACE_ATTR = {UHG_ATTR!r}")
unk = [k for k, r in RESULT.items() if r != "PASS"]
if unk:
    print("not enabled:", {k: RESULT[k] for k in unk})