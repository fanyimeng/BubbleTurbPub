import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bubturb import *
from astropy import units as u
from astropy.coordinates import SkyCoord
from tabulate import tabulate
from astropy.io import fits
from tqdm import tqdm
import os
import sys
from spectral_cube import SpectralCube
from concurrent.futures import ProcessPoolExecutor, as_completed

PVPLOT_DIR = "./pvplots"

# === Monkey patch for fwf export ===
def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
    open(fname, "w").write(content)

pd.DataFrame.to_fwf = to_fwf

# === Ellipse extractor ===
def est_ellipse(df, idx):
    ellipse = {}
    i = df['id'].tolist().index(idx)
    row = df.iloc[i]
    c = SkyCoord('%s  %s' % (row['ra_hms'], row['dec_dms']), unit=(u.hourangle, u.deg))
    ellipse['id'] = row['id']
    ellipse['ra'] = c.ra.value
    ellipse['dec'] = c.dec.value
    ellipse['major'] = row['maj_as']
    ellipse['minor'] = row['min_as']
    ellipse['pa'] = row['pa']
    ellipse['velocities'] = [row['vmin_kms'], row['vmax_kms']]
    ellipse['collide'] = row['collide']
    ellipse['v1'] = row['v1']
    ellipse['v2'] = row['v2']
    ellipse['v_ang'] = row['v_ang']
    return ellipse

# === Worker function ===
def process_one_bubble(idx, df, cube_path, datapath):
    try:
        ellipse = est_ellipse(df, idx)
        subcube, velocities = export_subcube_v3(
            SpectralCube.read(cube_path, format='fits'),
            ellipse,
            v_plot_factor=4.,
            scale_factor=3,
            output_file=f"{datapath}/pv-original-{ellipse['id']:03d}.fits"
        )
        # subcube, velocities = export_subcube_v3(cube, ellipse, scale_factor=3, v_plot_factor=2., output_file=None)
        subcube = SpectralCube.read(f"{datapath}/pv-original-{ellipse['id']:03d}.fits")
        bubble_pv_plot_v5(
            subcube, 38, plot_paths=True,
            # plot_name=f"t1+t2/pv-original-{ellipse['id']:03d}.pdf",
            plot_name=os.path.join(PVPLOT_DIR, f"pvPlot-{int(ellipse['id']):03d}.png"),
            ellipse=ellipse, reg_id=ellipse['id'],
            collide=ellipse['collide'],
            reg_file='b86_J2000.reg',
            plot_extra_ellipses=True
        )
        return f"Done ID {ellipse['id']}"
    except Exception as e:
        return f"Error ID {idx}: {e}"

# === Main ===
if __name__ == "__main__":
    plt.close("all")
    df = pd.read_fwf("0407-sorted.tab", header=0, infer_nrows=int(1e6))
    cube_path = '../data/jcomb_vcube_scaled2.fits'
    datapath = '/Users/meng/alex/astro/m31/00bubble/subcube_0407'

    # use all available CPUs
    max_workers = os.cpu_count()

    tasks = list(range(1, len(df) + 1))  # IDs start from 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_bubble, idx, df, cube_path, datapath) for idx in tasks]
        for f in tqdm(as_completed(futures), total=len(futures)):
            print(f.result())