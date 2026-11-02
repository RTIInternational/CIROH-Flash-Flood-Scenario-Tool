import numpy as np

def lhs(n, d, rng):
    """Latin hypercube in [0,1)^d, numpy only (no scipy needed in hydro-model env)."""
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        rng.shuffle(u[:, j])
    return u

def sample_members(base, cfg, n, seed=0):
    rng = np.random.default_rng(seed)
    mult_keys = list(cfg.get("mult", {}))
    u = lhs(n, 1 + len(mult_keys), rng)
    sat = cfg["sat"][0] + u[:, 0] * (cfg["sat"][1] - cfg["sat"][0])
    p = {k: np.full(n, float(v)) for k, v in base.items()}
    for j, k in enumerate(mult_keys, start=1):
        lo, hi = cfg["mult"][k]
        p[k] = base[k] * (lo + u[:, j] * (hi - lo))
    # keep SAC-SMA constraints: 0 < lzpk < lzsk < uzk < 1 ; pctim + adimp < 1
    p["uzk"]  = np.minimum(p["uzk"], 0.99)
    p["lzsk"] = np.minimum(p["lzsk"], p["uzk"] - 1e-3)
    p["lzpk"] = np.minimum(p["lzpk"], p["lzsk"] - 1e-4)
    over = p["pctim"] + p["adimp"] >= 1.0
    p["adimp"] = np.where(over, 0.99 - p["pctim"], p["adimp"])
    return sat, p

def states_from_saturation(sat, p, spin):
    uztwc = sat * p["uztwm"]
    lztwc = sat * p["lztwm"]
    uzfwc = np.clip(2 * sat - 1, 0, 1) * p["uzfwm"]
    lzfpc = np.minimum(spin["lzfpc"], p["lzfpm"])
    lzfsc = np.minimum(spin["lzfsc"], p["lzfsm"])
    return dict(uztwc=uztwc, uzfwc=uzfwc, lztwc=lztwc,
                lzfpc=lzfpc, lzfsc=lzfsc, adimc=uztwc + lztwc)

if __name__ == "__main__":
    base = dict(uztwm=44, uzfwm=21, lztwm=35, lzfpm=14, lzfsm=6, adimp=0.0, uzk=0.18,
                lzpk=0.001, lzsk=0.07, zperc=65, rexp=1.7, pctim=0.0, pfree=0.2)
    cfg = dict(sat=(0.3, 0.9), mult={"uzk": (0.7, 1.3), "uztwm": (0.8, 1.2),
                                     "zperc": (0.8, 1.2), "lzsk": (0.5, 2.0)})
    sat, p = sample_members(base, cfg, 1000, seed=1)
    st = states_from_saturation(sat, p, dict(lzfpc=5.0, lzfsc=2.0))
    print("sat range", sat.min().round(3), sat.max().round(3))
    print("order violations:", int(((p['lzpk'] >= p['lzsk']) | (p['lzsk'] >= p['uzk'])).sum()))
    print("capacity violations:", int(sum((st[s] > p[c] + 1e-9).sum() for s, c in
          [("uztwc","uztwm"),("uzfwc","uzfwm"),("lztwc","lztwm"),("lzfpc","lzfpm"),("lzfsc","lzfsm")])))
    # LHS stratification check: each 1/1000 bin of sat hit once
    print("strata filled:", len(np.unique(np.floor((sat-0.3)/0.6*1000).astype(int))))
