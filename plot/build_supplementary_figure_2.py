"""Generate and combine the final two PV examples."""
from paper_formula_common import REPO_ROOT, RUN_ROOT
from package_vertical import write_vertical
import subprocess
import sys
if __name__ == "__main__":
    out = RUN_ROOT / "pv_examples"
    subprocess.run([sys.executable, str(REPO_ROOT / "plot/plot_paper_pv_examples.py"),
        "--outdir", str(out), "--pp-cmap", "inferno", "--pv-cmap", "Greys_r",
        "--pv-norm", "linear", "--pv-clip-stat", "window_p05p95",
        "--pp-ellipse-color", "white", "--pp-cut-color", "1,1,1,0.9",
        "--pp-angle-label-color", "white", "--pp-angle-label-stroke-color", "none",
        "--savefig-dpi", "400"], check=True)
    write_vertical([out / "pv_examples_b10.pdf", out / "pv_examples_b17.pdf"],
        RUN_ROOT / "Supplementary_Figure_2.pdf", 180.0, 3.0)
