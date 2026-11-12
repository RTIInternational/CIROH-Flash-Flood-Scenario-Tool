"""
test_nhru.py - can pysacsnow SACSMA + UHG run many ensemble members as separate
HRUs (nhru = N)?  If not, how fast is the simple per-member loop instead?

Run from the code/ folder, in the *hydro-model* conda env (needs sac_worker.py and
ens_helpers.py in the same folder):

    python test_nhru.py                      # basin 16103000, N = 10
    python test_nhru.py 16010000 10          # another basin
    python test_nhru.py 16010000 10 --bench  # adds timing for 100 / 1000 members

Paste the SUMMARY block at the end back to Copilot.
"""
import sys
import json
import time
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pysacsnow as pss

_src_dir = (Path(__file__).parent.parent / "src").resolve()
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import sac_worker as sw          # reuse the worker's paths + spin-up

TEST_VERSION = "v3 (builds N-HRU controller from a temp YAML; has T6 + [diag])"
print("test_nhru_v3.py ->", TEST_VERSION)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
BASIN = args[0] if len(args) > 0 else "16103000"
N = int(args[1]) if len(args) > 1 else 10
BENCH = "--bench" in sys.argv
if N < 4 or N % 2:
    sys.exit("Use an even N >= 4")

EVENT_START = "2026-03-19T12:00:00"      # inside the forcing record
N_EVENT, N_EXTEND = 24, 12
N_TOTAL = N_EVENT + N_EXTEND
RTOL = 1e-6

SAC_P, UHG_P, FORCING, UHG_CTRL = sw._paths_for_basin(BASIN)
BASE_SAC = json.loads(Path(SAC_P).read_text())
BASE_UHG = json.loads(Path(UHG_P).read_text())
BASE = {k: float(v[0]) for k, v in BASE_SAC.items() if k != "hru_id"}

# synthetic 24-h storm (120 mm, Gaussian hyetograph) + 12 dry hours
_h = np.arange(N_EVENT)
_g = np.exp(-0.5 * ((_h - 10) / 3.0) ** 2)
PPT = np.concatenate([_g / _g.sum() * 120.0, np.zeros(N_EXTEND)])
TIMES = pd.date_range(EVENT_START, periods=N_TOTAL, freq="h")
_fdf = pd.read_csv(FORCING, parse_dates=["time"]).set_index("time")
PET = _fdf["hru_pet"].reindex(TIMES, method="nearest").to_numpy(float)

t0 = time.time()
SPIN, SPIN_DATE = sw._spinup(EVENT_START, SAC_P, FORCING)
print(f"basin {BASIN} | spin-up state date {SPIN_DATE} | spin-up {time.time()-t0:.1f}s")


# ── helpers ──────────────────────────────────────────────────────────────────
def _tmp(suffix):
    return Path(tempfile.mktemp(suffix=suffix))


def write_sac(n, **override):
    """Calibrated params replicated to n HRUs; override={name: scalar or length-n array}."""
    out = {}
    for k, v in BASE_SAC.items():
        out[k] = ([f"{v[0]}_{i}" for i in range(n)] if k == "hru_id"
                  else np.full(n, float(v[0])).tolist())
    for k, arr in override.items():
        out[k] = np.broadcast_to(np.asarray(arr, float), (n,)).tolist()
    p = _tmp(".json")
    p.write_text(json.dumps(out))
    return p


def write_uhg(n, ords=None):
    base = BASE_UHG["uh_ords"][0]
    rows = [list(base)] * n if ords is None else [[float(x) for x in r] for r in ords]
    out = {"hru_id": [f"{BASE_UHG['hru_id'][0]}_{i}" for i in range(n)],
           "hru_area": [BASE_UHG["hru_area"][0]] * n,
           "uh_ords": rows}
    p = _tmp(".json")
    p.write_text(json.dumps(out))
    return p


_DIAG_DONE = False


def _ctrl(template, params, n, trim):
    """
    Build a Controller from a temp YAML that already says size: n, instead of editing
    config.model.dimensions.size after loading (v1 of this test did that and the model
    arrays stayed size 1).  All paths are absolute; dates are trimmed to the event window
    for n > 1 so buffers don't hold ~5 years x n values.
    """
    import yaml
    cfg = yaml.safe_load(Path(template).read_text())
    cfg["model"]["dimensions"]["size"] = int(n)
    cfg["model"]["parameters"]["path"] = str(params)
    cfg["model"]["forcing"][0]["path"] = str(FORCING)
    cfg["simulation"]["output_path"] = str(_tmp(".csv"))
    if trim:
        t_start = pd.Timestamp(EVENT_START)
        cfg["simulation"]["start_date"] = t_start.strftime("%Y-%m-%d %H:%M:%S")
        cfg["simulation"]["end_date"] = (t_start + pd.Timedelta(hours=N_TOTAL + 1)).strftime("%Y-%m-%d %H:%M:%S")
    tmp = Path(template).parent / f"_tmp_ens_{next(tempfile._get_candidate_names())}.yaml"
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    try:
        return pss.Controller.load_from_yaml(tmp, verbose=False)
    finally:
        tmp.unlink(missing_ok=True)


def _shape(o, name):
    try:
        return np.shape(getattr(o, name))
    except Exception as e:
        return f"n/a ({type(e).__name__})"


def build(n, sac_params, uhg_params):
    global _DIAG_DONE
    if n == 1:                       # exactly what sac_worker does (this path passed T0 and T5)
        sc = pss.Controller.load_from_yaml(sw._SAC_CTRL_TEMPLATE, verbose=False)
        sc.config.model.parameters.path = sac_params
        sc.config.model.forcing[0].path = FORCING
        sc.config.simulation.output_path = _tmp(".csv")
        uc = pss.Controller.load_from_yaml(UHG_CTRL, verbose=False)
        uc.config.model.parameters.path = uhg_params
        uc.config.model.forcing[0].path = FORCING
        uc.config.simulation.output_path = _tmp(".csv")
    else:
        sc = _ctrl(sw._SAC_CTRL_TEMPLATE, sac_params, n, trim=True)
        uc = _ctrl(UHG_CTRL, uhg_params, n, trim=True)
    sac = pss.SACSMA(control=sc, calc_method="numpy")
    uhg = pss.UHG(control=uc, calc_method="numpy")
    if n != 1 and not _DIAG_DONE:    # one-off shape report to localise any size mismatch
        _DIAG_DONE = True
        print(f"  [diag] requested nhru={n}: SAC uzk{_shape(sac,'uzk')} uztwc{_shape(sac,'uztwc')} "
              f"hru_ppt{_shape(sac,'hru_ppt')} tci{_shape(sac,'tci')} | "
              f"UHG hru_tci{_shape(uhg,'hru_tci')} uh_ords{_shape(uhg,'uh_ords')} | "
              f"sac state buffer{np.shape(sac.buffer_states['uztwc'])}")
    return sc, uc, sac, uhg


def run_event(m, ppt, pet, states):
    """Same stepping loop as sac_worker._run_scenario, but returns (N_TOTAL, nhru)."""
    sc, uc, sac, uhg = m
    sac.reset()
    uhg.reset()
    for var, val in states.items():
        getattr(sac, var)[:] = np.asarray(val, dtype=np.float64)
    for i in range(N_TOTAL):
        sc.advance(); sac.advance()
        sac.hru_ppt[:] = ppt[i]
        sac.hru_pet[:] = pet[i]
        sac.hru_aesc[:] = 0.0
        sac.calculate(); sac.output()
        uc.advance(); uhg.advance()
        uhg.hru_tci[:] = sac.tci[:]
        uhg.calculate(); uhg.output()
    q = np.asarray(uhg.buffer_states["q"][:N_TOTAL], dtype=float)
    return q.reshape(N_TOTAL, -1)


RESULTS = []


def check(name, fn):
    try:
        status, msg = fn()
    except Exception as e:                      # keep going so one failure doesn't hide the rest
        status, msg = "FAIL", f"{type(e).__name__}: {e}"
        traceback.print_exc(limit=4)
    RESULTS.append((name, status, msg))
    print(f"[{status}] {name}: {msg}")
    return status


Q_REF = None


# ── tests ────────────────────────────────────────────────────────────────────
def t0_baseline():
    global Q_REF
    m = build(1, SAC_P, UHG_P)
    Q_REF = run_event(m, PPT, PET, SPIN)[:, 0]
    shp = np.shape(m[2].buffer_states["uztwc"])
    return "PASS", (f"single-HRU peak {Q_REF.max():.3f} m3/s at +{int(Q_REF.argmax())} h; "
                    f"state buffer shape {shp} (rows = control n_times)")


def t1_identical_members():
    m = build(N, write_sac(N), write_uhg(N))
    q = run_event(m, PPT, PET, SPIN)
    if q.shape != (N_TOTAL, N):
        return "FAIL", f"output shape {q.shape}, expected {(N_TOTAL, N)}"
    err = np.abs(q - Q_REF[:, None]).max() / Q_REF.max()
    return ("PASS" if err < RTOL else "FAIL"), \
        f"{N} identical members vs single run: max rel diff {err:.1e}"


def t2_param_and_crosstalk():
    half = N // 2
    px = np.r_[np.ones(half), np.full(N - half, 1.2)]
    m = build(N, write_sac(N, pxadj=px), write_uhg(N))
    q = run_event(m, PPT, PET, SPIN)
    lo = np.abs(q[:, :half] - Q_REF[:, None]).max() / Q_REF.max()
    hi_min = q[:, half:].max(axis=0).min()
    q12 = run_event(build(1, SAC_P, UHG_P), PPT * 1.2, PET, SPIN)[:, 0]
    as_precip = np.abs(q[:, half] - q12).max() / q12.max()
    if lo > RTOL:
        return "FAIL", f"members with pxadj=1 changed ({lo:.1e}) -> HRUs are leaking into each other"
    if hi_min <= Q_REF.max() * 1.0001:
        return "FAIL", "pxadj=1.2 had no effect on peak -> pxadj not applied inside calculate(); scale precip yourself"
    return "PASS", (f"pxadj=1.2 members peak >= {hi_min:.2f} vs {Q_REF.max():.2f}; no cross-talk; "
                    f"pxadj==precip*1.2 to {as_precip:.1e}")


def t3_states_per_member():
    from ens_helpers import states_from_saturation
    sat = np.linspace(0.2, 0.95, N)
    p = {k: np.full(N, v) for k, v in BASE.items()}
    st = states_from_saturation(sat, p, SPIN)
    m = build(N, write_sac(N), write_uhg(N))
    q = run_event(m, PPT, PET, st)
    pk = q.max(axis=0)
    mono = bool(np.all(np.diff(pk) >= -1e-9))
    flat = (pk[-1] - pk[0]) < 1e-3 * max(pk[-1], 1e-9)
    status = "FAIL" if not mono else ("WARN" if flat else "PASS")
    return status, (f"peak {pk[0]:.2f} -> {pk[-1]:.2f} m3/s as saturation 0.20 -> 0.95; "
                    f"monotonic={mono}" + ("; almost no sensitivity for this storm" if flat else ""))


def t4_uhg_per_member():
    from ens_helpers import scale_uh
    half = N // 2
    b = BASE_UHG["uh_ords"][0]
    ords = [b] * half + [scale_uh(b, 1.0)] * (N - half)
    q = run_event(build(N, write_sac(N), write_uhg(N, ords)), PPT, PET, SPIN)
    same = np.abs(q[:, 0] - Q_REF).max() / Q_REF.max()
    tp_a, tp_b = int(q[:, 0].argmax()), int(q[:, -1].argmax())
    dvol = abs(q[:, -1].sum() / q[:, 0].sum() - 1)
    ok = same < RTOL and tp_b >= tp_a and dvol < 0.05
    return ("PASS" if ok else "FAIL"), (f"per-member ordinates honoured: baseline match {same:.1e}, "
                                        f"time-to-peak {tp_a}h -> {tp_b}h, volume change {dvol:.1%}")


def t5_loop_path_inplace():
    new = dict(pxadj=1.2, uzk=min(0.99, BASE["uzk"] * 1.3), zperc=BASE["zperc"] * 0.8)
    qf = run_event(build(1, write_sac(1, **new), write_uhg(1)), PPT, PET, SPIN)[:, 0]
    m = build(1, SAC_P, UHG_P)
    for k, v in new.items():
        getattr(m[2], k)[:] = v
    qi = run_event(m, PPT, PET, SPIN)[:, 0]
    err = np.abs(qi - qf).max() / qf.max()
    moved = np.abs(qf - Q_REF).max() / Q_REF.max()
    for k in new:
        getattr(m[2], k)[:] = BASE[k]
    qb = run_event(m, PPT, PET, SPIN)[:, 0]
    back = np.abs(qb - Q_REF).max() / Q_REF.max()
    ok = err < RTOL and back < RTOL and moved > 1e-3
    return ("PASS" if ok else "FAIL"), (f"in-place == fresh build: {err:.1e}; parameters actually changed output: "
                                        f"{moved:.1%}; restore returns baseline: {back:.1e}")


def t6_uhg_inplace():
    """Loop path: can UHG ordinates be edited in place (needed for UHG-timing ensembles)?"""
    from ens_helpers import scale_uh
    base = np.asarray(BASE_UHG["uh_ords"][0], float)
    new = scale_uh(base, 1.0)
    q_fresh = run_event(build(1, SAC_P, write_uhg(1, [new])), PPT, PET, SPIN)[:, 0]
    m = build(1, SAC_P, UHG_P)
    uhg = m[3]
    hits = []
    for a in dir(uhg):
        if a.startswith("__"):
            continue
        try:
            v = getattr(uhg, a)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.size == base.size and np.allclose(v.ravel(), base):
            hits.append(a)
    if not hits:
        return "WARN", "no array attribute holds the ordinates -> for UHG timing, rewrite the UHG json and rebuild per member"
    for a in hits:
        getattr(uhg, a)[...] = new.reshape(getattr(uhg, a).shape)
    q_inpl = run_event(m, PPT, PET, SPIN)[:, 0]
    err = np.abs(q_inpl - q_fresh).max() / q_fresh.max()
    moved = np.abs(q_fresh - Q_REF).max() / Q_REF.max()
    ok = err < RTOL and moved > 1e-3
    return ("PASS" if ok else "WARN"), (f"attrs {hits}: in-place == fresh build {err:.1e}; shift changed output {moved:.1%}"
                                        + ("" if ok else " -> in-place edit not picked up; rebuild UHG per member"))


def bench_loop(M=20):
    rng = np.random.default_rng(0)
    m = build(1, SAC_P, UHG_P)
    t = time.time()
    for _ in range(M):
        m[2].pxadj[:] = BASE["pxadj"] * rng.uniform(0.9, 1.5)
        s = rng.uniform(0.3, 0.9)
        st = dict(SPIN, uztwc=s * BASE["uztwm"], lztwc=s * BASE["lztwm"])
        run_event(m, PPT, PET, st)
    per = (time.time() - t) / M
    return "INFO", f"loop path: {per*1000:.0f} ms/member -> 1000 members ~ {per*1000:.0f} s (extrapolated)"


def bench_vector(n):
    t = time.time()
    m = build(n, write_sac(n), write_uhg(n))
    tb = time.time() - t
    t = time.time()
    run_event(m, PPT, PET, SPIN)
    tr = time.time() - t
    mb = sum(getattr(v, "nbytes", 0) for v in m[2].buffer_states.values()) / 1e6
    return "INFO", f"nhru={n}: build {tb:.1f}s, run {tr:.1f}s, SAC buffers ~{mb:.0f} MB"


if check("T0 baseline single HRU", t0_baseline) == "PASS":
    s1 = check(f"T1 {N} identical HRUs == single run", t1_identical_members)
    if s1 == "PASS":
        check("T2 per-member parameter (pxadj) + no cross-talk", t2_param_and_crosstalk)
        check("T3 per-member initial states vs saturation", t3_states_per_member)
        check("T4 per-member UHG ordinates", t4_uhg_per_member)
    check("T5 loop path: in-place parameter edits == fresh build", t5_loop_path_inplace)
    check("T6 loop path: in-place UHG ordinate edit", t6_uhg_inplace)
    if BENCH:
        check("B1 loop timing", bench_loop)
        if s1 == "PASS":
            for n in (100, 1000):
                check(f"B2 vectorised timing n={n}", lambda n=n: bench_vector(n))

print("\n================ SUMMARY (paste this back) ================")
print(f"basin {BASIN}, N={N}, pysacsnow {getattr(pss, '__version__', 'unknown')}")
for name, status, msg in RESULTS:
    print(f"[{status}] {name}: {msg}")