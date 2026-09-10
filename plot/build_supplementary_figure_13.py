"""Generate and combine the three factor-decomposition panels."""
from paper_formula_common import REPO_ROOT, RUN_ROOT
from package_figure_4 import write_horizontal
import subprocess
import sys
if __name__ == "__main__":
    subprocess.run([sys.executable, str(REPO_ROOT / "plot/dot_e_ratio_panels_hi.py")], check=True)
    names = ["independence_edot_div_nhi.pdf", "independence_edot_div_v2.pdf", "independence_edot_times_time.pdf"]
    write_horizontal([RUN_ROOT / n for n in names], RUN_ROOT / "Supplementary_Figure_13.pdf", 58.8, 1.8)
