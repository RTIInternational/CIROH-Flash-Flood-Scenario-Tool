"""
ens_core2.py (v2) - ensemble sampling + driver for the SAC-SMA scenario tool.
Pure numpy; no pysacsnow import, so it can be tested without the model.
Used by sac_worker_ens2.py (runs in the 'hydro-model' env).
"""
import numpy as np

CORE_VERSION = "ens_core v2"
STATE_VARS = ("uztwc", "uzfwc", "lztwc", "lzfpc", "lzfsc", "adimc")

# a < b is enforced only if the CALIBRATED base set already satisfies it
# (basin 16010000 has lzsk > uzk, so that pair is skipped for it).
ORDER_PAIRS = [("lzpk", "lzsk"), ("lzsk", "uzk"), ("lzpk", "uzk")]
PCT = (5, 25, 50, 75, 95)


def lhs(n, d, rng):
    """Latin hypercube sample in [0,1)^d."""
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        rng.shuffle(u[:, j])
    return u


def sample_members(base, cfg, n, seed=0):
    """
    base : calibrated scalars {name: float}
    cfg  : {"sat": [lo, hi] | None,
            "mult": {param: [lo, hi]},      # multiplier on the calibrated value
            "add":  {param: [lo, hi]},      # additive offset (use when the base value is 0)
            "ppt_mult": [lo, hi],           # multiplier on the precipitation series
            "uh_shift": [lo, hi]}           # unit-hydrograph time shift, hours
    Returns (sat or None, params {name: array[n]}, extra {"ppt_mult": array, "uh_shift": array})
    """
    rng = np.random.default_rng(seed)
    mult, add = dict(cfg.get("mult") or {}), dict(cfg.get("add") or {})
    extra_ranges = {k: cfg[k] for k in ("ppt_mult", "uh_shift") if cfg.get(k)}
    sat_rng = cfg.get("sat")
    cols = (["sat"] if sat_rng else []) + list(mult) + list(add) + list(extra_ranges)
    u = lhs(n, max(1, len(cols)), rng)
    col = {name: u[:, j] for j, name in enumerate(cols)}

    p = {k: np.full(n, float(v)) for k, v in base.items()}
    for k, (lo, hi) in mult.items():
        p[k] = base[k] * (lo + col[k] * (hi - lo))
    for k, (lo, hi) in add.items():
        p[k] = base[k] + lo + col[k] * (hi - lo)
    for a, b in ORDER_PAIRS:
        if a in p and b in p and base[a] < base[b]:
            p[a] = np.minimum(p[a], p[b] * 0.999)
    p["uzk"] = np.minimum(p["uzk"], 0.99)
    over = p["pctim"] + p["adimp"] >= 1.0
    p["adimp"] = np.where(over, 0.99 - p["pctim"], p["adimp"])

    sat = (sat_rng[0] + col["sat"] * (sat_rng[1] - sat_rng[0])) if sat_rng else None
    extra = {k: lo + col[k] * (hi - lo) for k, (lo, hi) in extra_ranges.items()}
    return sat, p, extra


TENSION = (("uztwc", "uztwm"), ("lztwc", "lztwm"))
FREE = (("uzfwc", "uzfwm"), ("lzfpc", "lzfpm"), ("lzfsc", "lzfsm"))
_EPS = 1e-9


def saturation_of(states, params):
    """Tension-water saturation of a state dict: (UZTWC + LZTWC) / (UZTWM + LZTWM)."""
    return float((states["uztwc"] + states["lztwc"]) / (params["uztwm"] + params["lztwm"]))


def states_from_spinup(sat, p, spin, base):
    """
    Member initial states, scaled from the deterministic spin-up state.

    sat   : array (n,)  target tension-water saturation per member, in [0, 1]
    p     : dict of per-member parameter arrays (capacities may differ from `base`)
    spin  : spin-up state of the deterministic run (scalars)
    base  : calibrated parameters (scalars)

    Every store keeps the spin-up state's FRACTION of capacity, then is rescaled:
      * tension water (UZTWC, LZTWC): both zones are multiplied by one factor r, solved per member so
        that (UZTWC + LZTWC) / (UZTWM + LZTWM) equals `sat` exactly.  A zone that reaches capacity stays
        full and the other zone takes up the remainder.
      * free water (UZFWC, LZFPC, LZFSC) and ADIMC: multiplied by sat / (spin-up saturation).
      * every store is capped at the member's own capacity.
    With sat equal to the spin-up saturation and no parameter perturbation this returns the spin-up state
    exactly.  Parameter perturbation changes capacities, so the stores keep their fraction of the new capacity.
    """
    sat = np.asarray(sat, float)
    n = sat.size
    cap = {k: np.asarray(p[k], float) for _, k in TENSION + FREE}
    cap_t = cap["uztwm"] + cap["lztwm"]
    f = {c: np.clip(spin[c] / base[k], 0.0, 1.0) for c, k in TENSION + FREE}
    f_adimc = float(np.clip(spin["adimc"] / (base["uztwm"] + base["lztwm"]), 0.0, 1.0))

    # tension water: fractions, floored so a zone that is exactly empty in the spin-up can still be wetted
    ft = np.array([max(f["uztwc"], _EPS), max(f["lztwc"], _EPS)])
    s0_m = (ft[0] * cap["uztwm"] + ft[1] * cap["lztwm"]) / cap_t          # spin-up fractions on member capacities
    target = np.clip(sat, 0.0, 1.0) * cap_t

    lo, hi = np.zeros(n), np.full(n, 1.0 / ft.min())                       # hi: every zone certainly full
    for _ in range(80):                                                    # monotone -> bisection
        r = 0.5 * (lo + hi)
        got = cap["uztwm"] * np.minimum(1.0, r * ft[0]) + cap["lztwm"] * np.minimum(1.0, r * ft[1])
        low = got < target
        lo = np.where(low, r, lo)
        hi = np.where(low, hi, r)
    r = 0.5 * (lo + hi)
    # if the target is exactly reachable by the plain ratio (no zone full) use it, so identity is exact
    r_plain_t = np.clip(sat, 0.0, 1.0) / s0_m
    uncapped = np.maximum(r_plain_t * ft[0], r_plain_t * ft[1]) <= 1.0
    r = np.where(uncapped, r_plain_t, r)

    st = {"uztwc": cap["uztwm"] * np.minimum(1.0, r * ft[0]),
          "lztwc": cap["lztwm"] * np.minimum(1.0, r * ft[1])}
    s0 = max(saturation_of(spin, base), _EPS)
    r_free = np.clip(sat, 0.0, 1.0) / s0
    for c, k in FREE:
        st[c] = cap[k] * np.minimum(1.0, r_free * f[c])
    st["adimc"] = cap_t * np.minimum(1.0, r_free * f_adimc)
    return {v: st[v] for v in STATE_VARS}


def states_legacy(sat, p, spin):
    """The v1 rule (kept only for comparison, cfg {"state_method": "legacy"}): tension water = sat x capacity,
    upper free water only above 50 % saturation, lower free water held at the spin-up value."""
    sat = np.asarray(sat, float)
    uztwc = sat * p["uztwm"]
    lztwc = sat * p["lztwm"]
    uzfwc = np.clip(2 * sat - 1, 0, 1) * p["uzfwm"]
    lzfpc = np.minimum(spin["lzfpc"], p["lzfpm"])
    lzfsc = np.minimum(spin["lzfsc"], p["lzfsm"])
    return dict(uztwc=uztwc, uzfwc=uzfwc, lztwc=lztwc, lzfpc=lzfpc, lzfsc=lzfsc, adimc=uztwc + lztwc)


def scale_uh(uh_ords, shift_hr=0.0):
    """Shift a unit hydrograph by linear interpolation, renormalised to conserve volume."""
    uh = np.asarray(uh_ords, float)
    t = np.arange(len(uh))
    out = np.interp(t - shift_hr, t, uh, left=0.0, right=0.0)
    return out * (uh.sum() / out.sum()) if out.sum() > 0 else uh


def run_ensemble(cfg, base, spin, scenarios, run_fn, set_param, safe,
                 set_uh=None, uh_base=None, n_keep_traces=25):
    """
    run_fn(ppt_array, states_dict) -> 1-D flow array (m3/s), length n_total
    set_param(name, value)         -> edits a parameter of the live model in place
    safe                           -> parameter names whose in-place edit was verified
    Same member draws are used for every scenario, so scenario differences are not sampling noise.
    """
    n = int(cfg.get("n", 200))
    seed = int(cfg.get("seed", 0))
    perturbed = list((cfg.get("mult") or {})) + list((cfg.get("add") or {}))
    bad = [k for k in perturbed if k not in safe]
    if bad:
        raise ValueError(f"parameters not verified for in-place edits: {bad}. "
                         f"Run test_inplace.py and add them to _INPLACE_SAFE in sac_worker.py")
    if cfg.get("uh_shift") and (set_uh is None or uh_base is None):
        raise ValueError("uh_shift requested but _UHG_INPLACE_ATTR is not configured (run test_inplace.py)")
    if n < 2 or n > 5000:
        raise ValueError("n must be between 2 and 5000")

    method = cfg.get("state_method", "scaled")
    if method not in ("scaled", "legacy"):
        raise ValueError("state_method must be 'scaled' or 'legacy'")
    sat, p, extra = sample_members(base, cfg, n, seed)
    s_spin = saturation_of(spin, base)
    if sat is None:                                         # no saturation range: every member starts at the spin-up saturation
        sat = np.full(n, s_spin)
    st = states_from_spinup(sat, p, spin, base) if method == "scaled" else states_legacy(sat, p, spin)
    sat_report = sat
    ppt_mult = extra.get("ppt_mult", np.ones(n))
    uh_shift = extra.get("uh_shift", np.zeros(n))

    out = {}
    for label, ppt in scenarios.items():
        ppt = np.asarray(ppt, float)
        Q = None
        for i in range(n):
            for k in perturbed:
                set_param(k, float(p[k][i]))
            if set_uh is not None and uh_base is not None:
                set_uh(scale_uh(uh_base, float(uh_shift[i])))
            q = np.asarray(run_fn(ppt * ppt_mult[i], {v: float(st[v][i]) for v in STATE_VARS}), float)
            if Q is None:
                Q = np.empty((n, len(q)))
            Q[i] = q
        out[label] = {
            "pct": {str(q): np.percentile(Q, q, axis=0).tolist() for q in PCT},
            "peak": Q.max(axis=1).tolist(),                 # m3/s
            "tpeak_h": Q.argmax(axis=1).tolist(),           # hours after first time step
            "volume_m3": (Q.sum(axis=1) * 3600.0).tolist(),
            "traces": Q[:n_keep_traces].tolist(),
        }
    for k in perturbed:                                     # leave the live model as calibrated
        set_param(k, float(base[k]))
    if set_uh is not None and uh_base is not None:
        set_uh(np.asarray(uh_base, float))

    sat_achieved = (st["uztwc"] + st["lztwc"]) / (p["uztwm"] + p["lztwm"])
    members = {"saturation": sat_report.tolist(), "saturation_achieved": np.asarray(sat_achieved).tolist(), "ppt_mult": ppt_mult.tolist(), "uh_shift_h": uh_shift.tolist()}
    members.update({k: p[k].tolist() for k in perturbed})
    return {"meta": {"n": n, "seed": seed, "percentiles": list(PCT), "core": CORE_VERSION,
                     "sat_spinup": s_spin, "state_method": method, "config": cfg},
            "members": members, "scenarios": out}
