"""
ens_ui2.py (v2) - ensemble controls and plots for dashboard_mrms_ens.py.

* EnsControls   : Panel widgets + config() -> the "ensemble" block for sac_worker_ens.py
* summarize / summary_md : numpy-only data prep (no bokeh / panel needed)
* band_figure / sensitivity_figure / empty_figure : Bokeh drawing (bokeh imported lazily)

Imported by dashboard_mrms_ens.py (made by apply_ens_patch.py).  Lives next to utils.py.
"""
import numpy as np

UI_VERSION = "ens_ui v2"
PCT_KEYS = ("5", "25", "50", "75", "95")
# soil parameters sampled when "Soil parameters +/-20%" is ticked.  Chosen from the
# in-place test on basin 16010000 (largest hydrograph effect): all verified in _INPLACE_SAFE.
SOIL_PARAMS = ("uztwm", "lzfsm", "rexp", "uzfwm", "lztwm", "lzfpm")
SOIL_RANGE = (0.8, 1.2)
UH_SHIFT_H = 1.0


DESC = {
    "enable": "Off = one deterministic run per scenario, with no uncertainty bands.",
    "sat": ("How wet the soil is when the event starts, as the share of tension-water capacity that is full. "
            "Each member draws one value from this range."),
    "ppt": ("One factor per member that scales every hour of rainfall (1.0 = unchanged). "
            "Represents error in the rainfall amount, such as radar QPE bias."),
    "n": "Model runs per scenario. More members give smoother percentile bands; each run takes only a few milliseconds.",
    "soil": ("Varies six calibrated soil parameters (storage capacities and the percolation exponent) "
             "by \u00b120% per member. Represents calibration uncertainty."),
    "uh": ("Shifts the unit-hydrograph response earlier or later by up to \u00b11 h per member. "
           "Represents uncertainty in basin lag; changes timing and peak, not volume."),
}


def _desc(key):
    import panel as pn
    return pn.pane.Markdown(DESC[key], sizing_mode="stretch_width", min_width=300, margin=(0, 5, 0, 15),
                            styles={"font-size": "12px", "color": "#555"})


class EnsControls:
    """Ensemble settings shown above the SAC-SMA results (each control has a short description)."""

    def __init__(self):
        import panel as pn
        self.enable = pn.widgets.Checkbox(name="Run ensemble", value=True, width=260)
        self.n = pn.widgets.IntSlider(name="Members", start=50, end=1000, step=50, value=300, width=260)
        self.sat = pn.widgets.RangeSlider(name="Initial soil saturation (%)", start=0, end=100,
                                          step=5, value=(40, 100), width=260)
        self.ppt = pn.widgets.RangeSlider(name="Precipitation multiplier", start=0.5, end=2.0,
                                          step=0.05, value=(0.8, 1.2), width=260)
        self.soil = pn.widgets.Checkbox(name="Soil parameters \u00b120%", value=True, width=260)
        self.uh = pn.widgets.Checkbox(name="Unit-hydrograph timing \u00b11 h", value=False, width=260)

        def row(widget, key):
            return pn.Row(widget, _desc(key), sizing_mode="stretch_width", align="center")

        self._layout = pn.Column(
            pn.pane.Markdown("#### Ensemble (uncertainty) settings"),
            row(self.enable, "enable"),
            row(self.sat, "sat"),
            row(self.ppt, "ppt"),
            row(self.n, "n"),
            row(self.soil, "soil"),
            row(self.uh, "uh"),
            sizing_mode="stretch_width",
            styles={"background": "#eef2f7", "border-radius": "8px",
                    "padding": "6px 12px", "margin-top": "6px"},
        )

    def layout(self):
        return self._layout

    def config(self):
        """Dict for the worker's 'ensemble' key, or None when the ensemble is switched off."""
        if not self.enable.value:
            return None
        lo, hi = self.sat.value
        pl, ph = self.ppt.value
        cfg = {"n": int(self.n.value), "seed": 1,
               "sat": [float(lo) / 100.0, float(hi) / 100.0],
               "ppt_mult": [float(pl), float(ph)]}
        if self.soil.value:
            cfg["mult"] = {k: list(SOIL_RANGE) for k in SOIL_PARAMS}
        if self.uh.value:
            cfg["uh_shift"] = [-UH_SHIFT_H, UH_SHIFT_H]
        return cfg


# ── data prep (numpy only) ───────────────────────────────────────────────────
def summarize(ens, labels):
    """Per-scenario arrays from the worker's '_ensemble' block, in the order of `labels`."""
    out = []
    for lb in labels:
        s = ens["scenarios"].get(lb)
        if s is None:
            continue
        out.append({"label": lb,
                    "pct": {k: np.asarray(s["pct"][k], float) for k in PCT_KEYS},
                    "peak": np.asarray(s["peak"], float),
                    "tpeak": np.asarray(s["tpeak_h"], float)})
    return out


def summary_md(ens, runs, labels, fs, fu):
    """Markdown: what was sampled + peak-flow table.  Peaks are percentiles OVER MEMBERS."""
    meta = ens["meta"]
    cfg = meta.get("config", {}) or {}
    bits = [f"**{meta['n']} members** per scenario",
            f"model spin-up state at event start (the deterministic run): **{meta['sat_spinup'] * 100:.1f}%** "
            f"of tension-water capacity"]
    if cfg.get("sat"):
        bits.append(f"sampled saturation {cfg['sat'][0] * 100:.0f}\u2013{cfg['sat'][1] * 100:.0f}%")
    if cfg.get("ppt_mult"):
        bits.append(f"precipitation \u00d7{cfg['ppt_mult'][0]:.2f}\u2013{cfg['ppt_mult'][1]:.2f}")
    if cfg.get("mult"):
        lo, hi = next(iter(cfg["mult"].values()))
        bits.append(f"soil parameters \u00d7{lo:.2f}\u2013{hi:.2f}")
    if cfg.get("uh_shift"):
        bits.append(f"unit-hydrograph timing {cfg['uh_shift'][0]:+.0f} to {cfg['uh_shift'][1]:+.0f} h")
    lines = ["### Ensemble results", " \u00b7 ".join(bits), "",
             f"| Scenario | Deterministic peak ({fu}) | 5th | Median | 95th | Time to peak, 5\u201395% (h after start) |",
             "|---|---|---|---|---|---|"]
    for v in summarize(ens, labels):
        pk = v["peak"] * fs
        tp = v["tpeak"]
        det = float(np.max(runs[v["label"]])) * fs
        lines.append(f"| {v['label']} | {det:.2f} | {np.percentile(pk, 5):.2f} | {np.percentile(pk, 50):.2f} "
                     f"| {np.percentile(pk, 95):.2f} | {np.percentile(tp, 5):.0f}\u2013{np.percentile(tp, 95):.0f} |")
    return "\n".join(lines)


# ── bokeh drawing (bokeh imported lazily) ────────────────────────────────────
def empty_figure():
    from bokeh.plotting import figure
    return figure(height=40, sizing_mode="stretch_width")


def band_figure(view, det_runs, times, n_event, y_label, title, transform, colors, unit,
                x_range=None, height=280):
    """5-95% and 25-75% bands, median line and the deterministic run, per scenario.
    transform maps raw m3/s arrays -> display units (flow scale, or stage via rating curve)."""
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool, Span
    kw = dict(x_range=x_range) if x_range is not None else {}
    p = figure(x_axis_type="datetime", height=height, sizing_mode="stretch_width", title=title,
               x_axis_label="Time (UTC)", y_axis_label=y_label, **kw)
    p.grid.grid_line_alpha = 0.3
    p.grid.grid_line_color = "gray"
    times = list(times)
    for i, v in enumerate(view):
        c = colors[i % len(colors)]
        y = {k: np.asarray(transform(v["pct"][k]), float) for k in PCT_KEYS}
        det = np.asarray(transform(det_runs[v["label"]]), float)
        m = min(len(times), len(det), *(len(a) for a in y.values()))
        data = {"x": times[:m], "det": list(det[:m]),
                "dt": [t.strftime("%Y-%m-%d %H:%Mz") for t in times[:m]]}
        data.update({f"p{k}": list(y[k][:m]) for k in PCT_KEYS})
        src = ColumnDataSource(data=data)
        p.varea(x="x", y1="p5", y2="p95", source=src, fill_color=c, fill_alpha=0.12,
                legend_label=f"{v['label']} 5\u201395%")
        p.varea(x="x", y1="p25", y2="p75", source=src, fill_color=c, fill_alpha=0.25)
        med = p.line("x", "p50", source=src, color=c, line_width=2,
                     legend_label=f"{v['label']} median")
        p.line("x", "det", source=src, color=c, line_width=1.5, line_dash="dashed",
               legend_label=f"{v['label']} deterministic")
        p.add_tools(HoverTool(renderers=[med], mode="vline", tooltips=[
            ("Time", "@dt"), ("5\u201395%", f"@p5{{0.00}} \u2013 @p95{{0.00}} {unit}"),
            ("25\u201375%", f"@p25{{0.00}} \u2013 @p75{{0.00}} {unit}"),
            ("Median", f"@p50{{0.00}} {unit}"), ("Deterministic", f"@det{{0.00}} {unit}")]))
    n_ext = len(times) - n_event
    if n_ext > 0:
        p.add_layout(Span(location=times[n_event - 1].timestamp() * 1000, dimension="height",
                          line_color="#888888", line_dash="dashed", line_width=1.5, line_alpha=0.8))
    p.legend.click_policy = "hide"
    p.add_layout(p.legend[0], "right")
    return p


def _scatter_fig(x, view, fs, fu, colors, title, x_label, vline, x_range, height):
    """Peak flow of every member vs one sampled variable, with a dashed reference line."""
    from bokeh.plotting import figure
    from bokeh.models import Span
    p = figure(height=height, sizing_mode="stretch_width", x_range=x_range, title=title,
               x_axis_label=x_label, y_axis_label=f"Peak streamflow ({fu})")
    p.grid.grid_line_alpha = 0.3
    for i, v in enumerate(view):
        p.scatter(list(x), list(v["peak"] * fs), color=colors[i % len(colors)], alpha=0.35,
                  size=5, legend_label=v["label"])
    p.add_layout(Span(location=float(vline), dimension="height",
                      line_color="#444444", line_dash="dashed", line_width=1.5))
    p.legend.click_policy = "hide"
    p.add_layout(p.legend[0], "right")
    return p


def sensitivity_figure(ens, view, fs, fu, colors, height=240):
    """Peak flow vs (1) initial soil saturation and (2) precipitation multiplier.
    A plot is skipped when its variable was not varied (all members share one value).
    Returns one figure, a column of figures, or an empty placeholder."""
    members = ens["members"]
    n = int(ens["meta"]["n"])
    figs = []
    sat = np.asarray(members["saturation"], float) * 100.0
    if np.ptp(sat) > 1e-9:
        figs.append(_scatter_fig(
            sat, view, fs, fu, colors,
            "Peak flow vs. initial soil saturation (dashed line = model spin-up state at event start)",
            "Initial tension-water saturation (%)", float(ens["meta"]["sat_spinup"]) * 100.0,
            (0, 100), height))
    pm = np.asarray(members.get("ppt_mult", np.ones(n)), float)
    if np.ptp(pm) > 1e-9:
        lo, hi = min(float(pm.min()), 1.0), max(float(pm.max()), 1.0)
        pad = 0.04 * (hi - lo)
        figs.append(_scatter_fig(
            pm, view, fs, fu, colors,
            "Peak flow vs. precipitation multiplier (dashed line = rainfall unchanged, \u00d71.0)",
            "Precipitation multiplier (\u00d7)", 1.0, (lo - pad, hi + pad), height))
    if not figs:
        return empty_figure()
    if len(figs) == 1:
        return figs[0]
    from bokeh.layouts import column
    return column(*figs, sizing_mode="stretch_width")
