"""Figures for the README, rendered in a light and a dark variant.

Style rules: thin marks (2 px lines, 8 px markers with a 2 px surface ring),
hairline solid gridlines, one y-axis per chart, a legend whenever there are two
or more series, text in ink colours (never the series colour), and a
colour-blind-validated categorical palette (blue = Heston / CPU, orange =
Bates / GPU). Sizes are in CSS pixels for a 200-dpi image shown at 50%.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

from .black_scholes import implied_vol  # noqa: E402
from .cos import cos_price  # noqa: E402
from .svi import svi_surface  # noqa: E402

DPI = 200
PX = 0.72  # one CSS pixel in points

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"),
                  seq=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"),
                 seq=("#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb")),
}


def _font_family():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Segoe UI", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"):
        if name in available:
            return name
    return "sans-serif"


def _rc(theme: str) -> dict:
    t = THEMES[theme]
    return {
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["axis"], "axes.linewidth": PX, "axes.labelcolor": t["ink2"],
        "axes.titlecolor": t["ink"], "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlesize": 13 * PX, "axes.titlepad": 8, "axes.labelsize": 12 * PX,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": t["grid"], "grid.linewidth": PX,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": t["axis"], "ytick.color": t["axis"], "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"], "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.minor.size": 0, "ytick.minor.size": 0, "xtick.labelsize": 11 * PX,
        "ytick.labelsize": 11 * PX, "text.color": t["ink"], "font.family": _font_family(),
        "font.size": 12 * PX, "legend.fontsize": 11 * PX, "legend.frameon": False,
        "lines.linewidth": 2 * PX, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 8 * PX, "lines.markeredgewidth": 2 * PX,
    }


def _header(fig, title: str, subtitle: str, theme: str, legend_handles=None):
    """Title + subtitle (+ optional legend on the right) placed above all axes."""
    t = THEMES[theme]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
    boxes = [ax.get_tightbbox(r) for ax in fig.axes if ax.get_visible()]
    x0 = min(b.x0 for b in boxes) / W
    x1 = max(b.x1 for b in boxes) / W
    y = max(b.y1 for b in boxes) / H
    fig.text(x0, y, subtitle, ha="left", va="bottom", fontsize=12 * PX, color=t["ink2"],
             transform=offset_copy(fig.transFigure, fig, y=12 * PX, units="points"))
    fig.text(x0, y, title, ha="left", va="bottom", fontsize=15 * PX, fontweight="semibold",
             color=t["ink"], transform=offset_copy(fig.transFigure, fig, y=34 * PX, units="points"))
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower right", ncol=len(legend_handles),
                   bbox_to_anchor=(x1, y), borderaxespad=0.0, handlelength=1.6, columnspacing=1.4,
                   bbox_transform=offset_copy(fig.transFigure, fig, y=10 * PX, units="points"))


def _compact(n, _pos=None):
    n = float(n)
    for div, suf in ((2**20, "M"), (2**10, "K")):
        if n >= div:
            v = n / div
            return f"{v:.0f}{suf}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}{suf}"
    return f"{n:.0f}"


def _save(fig, outdir: Path, name: str, theme: str, tight: bool = True) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}_{theme}.png"
    if tight:
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.18)
    else:
        fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _rounded_bar(ax, x, h, w, color, fig, radius_px=4):
    """Vertical bar with a 4 px rounded data-end and a square baseline (one path, no seams)."""
    if not np.isfinite(h) or h <= 0:
        return
    pos = ax.get_position()
    ax_w = fig.get_figwidth() * DPI * pos.width
    ax_h = fig.get_figheight() * DPI * pos.height
    (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
    rx = min(radius_px * 2 * (xb - xa) / ax_w, w / 2)
    ry = min(radius_px * 2 * (yb - ya) / ax_h, h)
    k = 0.5523
    l, rgt = x - w / 2, x + w / 2
    verts = [(l, 0), (l, h - ry), (l, h - ry + k * ry), (l + rx - k * rx, h), (l + rx, h),
             (rgt - rx, h), (rgt - rx + k * rx, h), (rgt, h - ry + k * ry), (rgt, h - ry),
             (rgt, 0), (l, 0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, linewidth=0, zorder=3))


# ---------------------------------------------------------------- surface
def plot_vol_surface(fits, slices, outdir: Path, themes=("light", "dark")):
    fits = sorted(fits, key=lambda f: f.T)
    z_grid = np.linspace(-0.65, 0.25, 90)          # ln(K/F) / sqrt(T)
    Ts = np.array([f.T for f in fits])
    Z = np.full((len(fits), z_grid.size), np.nan)
    for i, f in enumerate(fits):
        k = z_grid * np.sqrt(f.T)
        row = svi_surface([f], k)[0]
        Z[i] = row
    X, Y = np.meshgrid(z_grid, Ts)
    out = []
    for theme in themes:
        t = THEMES[theme]
        with plt.rc_context(_rc(theme)):
            fig = plt.figure(figsize=(8.6, 5.6))
            # 3-D axes are padded internally; oversize them so the surface fills the canvas
            ax = fig.add_axes([-0.04, -0.07, 0.90, 1.10], projection="3d", computed_zorder=False)
            cmap = LinearSegmentedColormap.from_list("blue_ramp", t["seq"])
            surf = ax.plot_surface(X, Y, 100 * Z, cmap=cmap, linewidth=0, antialiased=True,
                                   rstride=1, cstride=1, alpha=0.95, zorder=1)
            for s in slices:
                zz = s.k / np.sqrt(s.T)
                m = (zz >= z_grid[0]) & (zz <= z_grid[-1])
                ax.scatter(zz[m], np.full(m.sum(), s.T), 100 * s.iv_mid[m], s=1.2, color=t["ink"],
                           alpha=0.45, depthshade=False, zorder=2)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color(matplotlib.colors.to_rgba(t["surface"], 1.0))
                axis._axinfo["grid"]["color"] = t["grid"]
                axis._axinfo["grid"]["linewidth"] = PX
                axis.line.set_color(t["axis"])
            ax.set_xlabel("ln(K/F) / sqrt(T)", labelpad=6)
            ax.set_ylabel("maturity (years)", labelpad=6)
            ax.set_zlabel("implied vol (%)", labelpad=4)
            ax.tick_params(colors=t["ink2"], labelsize=10 * PX, pad=1)
            ax.view_init(elev=24, azim=-58)
            ax.set_box_aspect((1.3, 1.0, 0.72), zoom=1.0)
            cax = fig.add_axes([0.915, 0.27, 0.013, 0.40])
            cb = fig.colorbar(surf, cax=cax)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=10 * PX, colors=t["ink2"], length=0)
            fig.text(0.035, 0.955, "Implied-volatility surface", ha="left", va="top",
                     fontsize=15 * PX, fontweight="semibold", color=t["ink"])
            fig.text(0.035, 0.905, "SVI fit per expiry (surface) and market mid quotes (dots); "
                     "colour = implied vol (%)", ha="left", va="top", fontsize=12 * PX, color=t["ink2"])
            out.append(_save(fig, outdir, "vol_surface", theme, tight=False))
    return out


# ---------------------------------------------------------------- smiles
def plot_smiles(slices, heston, bates, outdir: Path, n_panels: int = 6, themes=("light", "dark")):
    slices = sorted(slices, key=lambda s: s.T)
    idx = np.unique(np.round(np.linspace(0, len(slices) - 1, min(n_panels, len(slices)))).astype(int))
    chosen = [slices[i] for i in idx]
    curves = []
    for s in chosen:
        k = np.linspace(s.k.min(), s.k.max(), 160)
        K = s.F * np.exp(k)
        calls = K >= s.F
        ivs = {}
        for name, p in (("Heston", heston), ("Bates", bates)):
            price = cos_price(p.log_cf, s.F, K, s.T, s.D, calls)
            ivs[name] = 100 * implied_vol(price, s.F, K, s.T, s.D, calls)
        curves.append((k, ivs))
    out = []
    ncol = 3
    nrow = int(np.ceil(len(chosen) / ncol))
    for theme in themes:
        t = THEMES[theme]
        with plt.rc_context(_rc(theme)):
            fig, axes = plt.subplots(nrow, ncol, figsize=(10.0, 3.0 * nrow + 0.4), squeeze=False)
            for ax, s, (k, ivs) in zip(axes.flat, chosen, curves):
                lo = 100 * np.where(np.isfinite(s.iv_bid), s.iv_bid, s.iv_mid)
                hi = 100 * np.where(np.isfinite(s.iv_ask), s.iv_ask, s.iv_mid)
                ax.vlines(s.k, lo, hi, color=t["muted"], linewidth=1.5 * PX, alpha=0.8, zorder=2)
                ax.plot(s.k, 100 * s.iv_mid, "o", markersize=2.5 * PX, color=t["ink2"],
                        markeredgewidth=0, zorder=3)
                ax.plot(k, ivs["Heston"], color=t["series"][0], zorder=4)
                ax.plot(k, ivs["Bates"], color=t["series"][1], zorder=5)
                ax.set_title(f"{s.dte:.0f} days  ({s.expiry})", fontsize=12 * PX)
                ax.set_xlabel("log-moneyness ln(K/F)")
                ax.set_ylabel("implied vol (%)")
            for ax in list(axes.flat)[len(chosen):]:
                ax.set_visible(False)
            handles = [Line2D([0], [0], color=t["muted"], linewidth=1.5 * PX, marker="o",
                              markersize=2.5 * PX, markerfacecolor=t["ink2"], markeredgewidth=0,
                              label="Market (bid-ask, mid)"),
                       Line2D([0], [0], color=t["series"][0], label="Heston"),
                       Line2D([0], [0], color=t["series"][1], label="Bates (Heston + jumps)")]
            fig.tight_layout(h_pad=1.6, w_pad=1.2)
            _header(fig, "Calibrated smiles vs market quotes",
                    "One calibration across all expiries; vertical bars are the bid-ask range", theme,
                    legend_handles=handles)
            out.append(_save(fig, outdir, "smile_fits", theme))
    return out


# ---------------------------------------------------------------- errors
def plot_error_buckets(err: pd.DataFrame, outdir: Path, themes=("light", "dark")):
    cross = err[err["group"] == "Maturity x Moneyness"].copy()
    cross[["mat", "mny"]] = cross["bucket"].str.split(r" \| ", expand=True)
    mats = ["<= 30d", "31-90d", "> 90d"]
    panels = ["OTM puts (<25 delta)", "Near ATM (25-50 delta)", "OTM calls (<25 delta)"]
    titles = {"OTM puts (<25 delta)": "OTM puts (|delta| < 0.25)",
              "Near ATM (25-50 delta)": "Near the money (0.25-0.5 delta)",
              "OTM calls (<25 delta)": "OTM calls (delta < 0.25)"}
    ymax = np.nanmax(cross["rmse"].values) * 1.18
    out = []
    for theme in themes:
        t = THEMES[theme]
        with plt.rc_context(_rc(theme)):
            fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), sharey=True)
            fig.tight_layout(w_pad=1.5)
            for ax, pan in zip(axes, panels):
                ax.set_xlim(-0.6, len(mats) - 0.4)
                ax.set_ylim(0, ymax)
                ax.grid(axis="x", visible=False)
                ax.set_xticks(range(len(mats)))
                ax.set_xticklabels([m.replace("<=", "≤") for m in mats])
                ax.set_title(titles[pan], fontsize=12 * PX)
                w = 0.27
                for j, model in enumerate(("Heston", "Bates")):
                    sub = cross[(cross["model"] == model) & (cross["mny"] == pan)].set_index("mat")
                    for i, m in enumerate(mats):
                        if m in sub.index:
                            h = float(sub.loc[m, "rmse"])
                            x = i + (j - 0.5) * (w + 0.03)
                            _rounded_bar(ax, x, h, w, t["series"][j], fig)
                            if pan == panels[0]:
                                ax.text(x, h + ymax * 0.015, f"{h:.2f}", ha="center", va="bottom",
                                        fontsize=10 * PX, color=t["ink2"])
            axes[0].set_ylabel("implied-vol RMSE (vol points)")
            handles = [Rectangle((0, 0), 1, 1, color=t["series"][0], label="Heston"),
                       Rectangle((0, 0), 1, 1, color=t["series"][1], label="Bates (Heston + jumps)")]
            _header(fig, "Where the models miss", "Implied-vol RMSE by maturity and moneyness, "
                    "all quotes after cleaning", theme, legend_handles=handles)
            out.append(_save(fig, outdir, "error_buckets", theme))
    return out


# ---------------------------------------------------------------- MC speed
def plot_mc_benchmark(bench: pd.DataFrame, outdir: Path, themes=("light", "dark")):
    order = ["NumPy float64 (CPU)", "NumPy float32 (CPU)", "CuPy array ops float32 (GPU)",
             "Fused CUDA kernel float32 (GPU)", "Fused CUDA kernel float64 (GPU)"]
    labels = [l for l in order if l in set(bench["label"])]
    out = []
    for theme in themes:
        t = THEMES[theme]
        with plt.rc_context(_rc(theme)):
            fig, ax = plt.subplots(figsize=(8.0, 4.4))
            for i, lab in enumerate(labels):
                d = bench[bench["label"] == lab].sort_values("n_paths")
                ax.plot(d["n_paths"], d["seconds"], "-o", color=t["series"][i], label=lab,
                        markeredgecolor=t["surface"], zorder=3 + i)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(_compact))
            ax.set_xlabel(f"number of paths ({int(bench['n_steps'].iloc[0])} time steps each)")
            ax.set_ylabel("simulation time (seconds, log scale)")
            ax.legend(loc="upper left")
            base = bench[bench["label"] == "NumPy float64 (CPU)"].set_index("n_paths")["seconds"]
            base32 = bench[bench["label"] == "NumPy float32 (CPU)"].set_index("n_paths")["seconds"]
            fast = bench[bench["label"] == "Fused CUDA kernel float32 (GPU)"].set_index("n_paths")["seconds"]
            common = sorted(set(base.index) & set(base32.index) & set(fast.index))
            if common:
                n = common[-1]
                ax.text(0.98, 0.04, f"At {_compact(n)} paths the fused float32 kernel is\n"
                        f"{base32[n] / fast[n]:,.0f}x faster than NumPy float32 (same precision)\n"
                        f"{base[n] / fast[n]:,.0f}x faster than NumPy float64",
                        transform=ax.transAxes, ha="right", va="bottom", color=t["ink"],
                        fontsize=11 * PX, linespacing=1.5)
            _header(fig, "Heston QE Monte Carlo: CPU vs GPU",
                    "Same algorithm and random-number budget; GPU timings include device sync", theme)
            out.append(_save(fig, outdir, "mc_benchmark", theme))
    return out


# ---------------------------------------------------------------- convergence
def plot_convergence(conv: pd.DataFrame, outdir: Path, themes=("light", "dark")):
    out = []
    ref = float(conv["cos_price"].iloc[0])
    methods = ["Plain + antithetic", "Conditional + antithetic"]
    for theme in themes:
        t = THEMES[theme]
        with plt.rc_context(_rc(theme)):
            fig, ax = plt.subplots(figsize=(8.0, 4.2))
            ax.axhline(ref, color=t["ink2"], linewidth=PX, zorder=2)
            for i, m in enumerate(methods):
                d = conv[conv["method"] == m].sort_values("n_paths")
                col = t["series"][i]
                ax.fill_between(d["n_paths"], d["price"] - 1.96 * d["stderr"], d["price"] + 1.96 * d["stderr"],
                                color=col, alpha=0.10, linewidth=0, zorder=1)
                ax.plot(d["n_paths"], d["price"], "-o", color=col, label=f"{m} (95% CI band)",
                        markeredgecolor=t["surface"], zorder=3 + i)
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(FuncFormatter(_compact))
            ax.set_xlabel("number of paths")
            ax.set_ylabel("ATM call price (index points)")
            ax.annotate("COS (semi-analytic) price", xy=(conv["n_paths"].max(), ref), xytext=(0, -6),
                        textcoords="offset points", va="top", ha="right", color=t["ink2"],
                        fontsize=11 * PX)
            ax.legend(loc="upper right")
            _header(fig, "Conditional Monte Carlo converges far faster",
                    "Estimate and 95% confidence band vs number of paths; calibrated Heston, 1-year ATM call",
                    theme)
            out.append(_save(fig, outdir, "mc_convergence", theme))
    return out
