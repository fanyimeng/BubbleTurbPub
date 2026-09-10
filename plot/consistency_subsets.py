"""Redraw the final FM/RL consistency diagnostic from its frozen plotting tables."""
from paper_formula_common import REPO_ROOT, RUN_ROOT
import numpy as np
import pandas as pd
import dot_e_balance as energy

def _filled_handle(energy_module, color: str, label: str):
    return energy_module.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=6.5,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.45,
        alpha=0.82,
        label=label,
    )

def _open_cross_handle(energy_module, color: str):
    circle = energy_module.Line2D(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=7.5,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=1.15,
    )
    cross = energy_module.Line2D(
        [],
        [],
        marker="x",
        linestyle="None",
        markersize=5.5,
        color=color,
        markeredgewidth=1.05,
    )
    return (circle, cross)

def draw_panel(ax, df: pd.DataFrame, panel_label: str, energy_module, show_ylabel: bool) -> dict[str, int]:
    energy_module.draw_turb_band(
        ax,
        turb_table_path=energy_module.DEFAULT_TURB_TABLE,
        prof_max_path=None,
        prof_min_path=None,
        turb_xlim=(3.0, 33.0),
        color=(0.12, 0.12, 0.12, 0.26),
        edgecolor=(0.0, 0.0, 0.0, 0.0),
        linewidth=0.0,
        zorder=1,
        label="Turbulence",
    )

    styles = {
        "shared yes": {
            "color": "#cf5f58",
            "label": "shared yes",
            "zorder": 8,
        },
        "catalogue-only yes": {
            "color": "#2f77b4",
            "label": "catalogue-only yes",
            "zorder": 10,
        },
        "independent-only yes": {
            "color": "#d8902f",
            "label": "independent-only yes",
            "zorder": 10,
        },
    }

    counts: dict[str, int] = {}
    for plot_class, style in styles.items():
        for included_here in (True, False):
            sub = df[
                (df["plot_class"] == plot_class)
                & (df["included_here"].eq(included_here))
                & df["finite_plot"]
            ]
            key = f"{plot_class} ({'included' if included_here else 'not selected here'})"
            counts[key] = int(len(sub))
            if sub.empty:
                continue
            color = style["color"]
            alpha = 0.28 if included_here else 0.22
            ax.errorbar(
                sub["R_kpc_plot"],
                sub["dotE_1e-28_plot"],
                xerr=sub["xerr_kpc_plot"],
                yerr=sub["yerr_1e-28_plot"],
                fmt="none",
                ecolor=color,
                alpha=alpha,
                elinewidth=0.75,
                capsize=0,
                zorder=style["zorder"] - 1,
            )
            if included_here:
                ax.scatter(
                    sub["R_kpc_plot"],
                    sub["dotE_1e-28_plot"],
                    s=58,
                    marker="o",
                    facecolors=color,
                    edgecolors="white",
                    linewidths=0.45,
                    alpha=0.82,
                    label=style["label"],
                    zorder=style["zorder"],
                )
            else:
                ax.scatter(
                    sub["R_kpc_plot"],
                    sub["dotE_1e-28_plot"],
                    s=82,
                    marker="o",
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.25,
                    alpha=0.95,
                    label=f"{style['label']} (not selected here)",
                    zorder=style["zorder"] + 1,
                )
                ax.scatter(
                    sub["R_kpc_plot"],
                    sub["dotE_1e-28_plot"],
                    s=28,
                    marker="x",
                    c=color,
                    linewidths=1.05,
                    alpha=0.95,
                    zorder=style["zorder"] + 2,
                )

    ax.text(
        0.04,
        0.94,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.6),
        zorder=30,
    )
    ax.set_yscale("log")
    ax.set_xlim(4.0, 28.0)
    ax.set_ylim(1e-2, 3e2)
    ax.set_xlabel("Distance to M31's center: $R$ [kpc]")
    if show_ylabel:
        ax.set_ylabel(r"$\dot{e}$ [$10^{-28}$ erg cm$^{-3}$ s$^{-1}$]")
    ax.tick_params(direction="in", which="both", length=5, top=True, bottom=True, left=True, right=True)
    ax.yaxis.set_minor_locator(energy_module.mticker.NullLocator())
    ax.set_box_aspect(1.0)

    colors = {key: value["color"] for key, value in styles.items()}
    handles = [
        energy_module.Rectangle((0, 0), 1, 1, facecolor=(0.12, 0.12, 0.12, 0.26), edgecolor="none"),
        _filled_handle(energy_module, colors["shared yes"], "FM+RL"),
    ]
    labels = ["turb.", "FM+RL"]
    if panel_label == "FM":
        handles.extend(
            [
                _filled_handle(energy_module, colors["catalogue-only yes"], "FM only"),
                _open_cross_handle(energy_module, colors["independent-only yes"]),
            ]
        )
        labels.extend(["FM only", "RL only"])
    else:
        handles.extend(
            [
                _open_cross_handle(energy_module, colors["catalogue-only yes"]),
                _filled_handle(energy_module, colors["independent-only yes"], "RL only"),
            ]
        )
        labels.extend(["FM only", "RL only"])
    legend = ax.legend(
        handles,
        labels,
        handler_map={tuple: energy_module.HandlerTuple(ndivide=1)},
        loc="upper right",
        fontsize=7.0,
        frameon=True,
        framealpha=0.88,
        borderpad=0.35,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    legend.get_frame().set_linewidth(0.4)
    legend.get_frame().set_edgecolor("0.4")
    legend.set_zorder(35)
    return counts

if __name__ == "__main__":
    plt = energy.plt
    catalogue_panel = pd.read_csv(REPO_ROOT / "data/consistency_fm_points.tsv", sep="\t")
    independent_panel = pd.read_csv(REPO_ROOT / "data/consistency_rl_points.tsv", sep="\t")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.25), sharex=True, sharey=True)
    counts_truth = draw_panel(
        axes[0],
        catalogue_panel,
        "FM",
        energy,
        show_ylabel=True,
    )
    counts_colleague = draw_panel(
        axes[1],
        independent_panel,
        "RL",
        energy,
        show_ylabel=False,
    )

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.98, wspace=0.08)

    pdf_path = RUN_ROOT / "Supplementary_Figure_3.pdf"
    png_path = RUN_ROOT / "Supplementary_Figure_3.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
