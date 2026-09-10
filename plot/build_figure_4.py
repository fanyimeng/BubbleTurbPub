"""Build final Figure 4 from the two energy-rate comparison panels."""
from paper_formula_common import RUN_ROOT
import matplotlib as mpl
from dot_e_vs_turb import plot_dot_e_vs_turb
from dot_e_vs_timescale_insets import plot_dot_e_vs_timescale
from package_figure_4 import write_horizontal

if __name__ == "__main__":
    mpl.rcParams["pdf.fonttype"] = 42
    a = RUN_ROOT / "edot_pair_panel_a_dot_e_vs_turb.pdf"
    b = RUN_ROOT / "edot_pair_panel_b_dot_e_vs_timescale.pdf"
    plot_dot_e_vs_turb(output_pdf=a)
    plot_dot_e_vs_timescale(output_pdf=b)
    write_horizontal([a, b], RUN_ROOT / "Figure_4.pdf", 89.0, 2.0)
