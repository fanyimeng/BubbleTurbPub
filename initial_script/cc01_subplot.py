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


def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
    open(fname, "w").write(content)


pd.DataFrame.to_fwf = to_fwf


def est_ellipse(df, idx):
    # 'id',  'ra_hms'   ,  'dec_dms'      major    minor    pa    v_max    v_min  in_b86

    ellipse = {}
    i = df['id'].tolist().index(idx)

    row = df.iloc[i]
    ellipse['id'] = row['id']
    c = SkyCoord('%s  %s' %
                 (row['ra_hms'], row['dec_dms']), unit=(u.hourangle, u.deg))
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


plt.close("all")

df = pd.read_fwf("0406-sorted.tab", header=0, infer_nrows=int(1e6))

# print(est_ellipse(df, 64))


header = fits.open('../data/jcomb_vcube_scaled2.fits')[0].header
cube = SpectralCube.read('../data/jcomb_vcube_scaled2.fits', format='fits')
datapath = '/Users/meng/alex/astro/m31/00bubble/subcube_0406'

for i in tqdm(range(0,len(df))):
    i = i+1
    ellipse = est_ellipse(df, i)
    subcube, velocities = export_subcube_v2(cube, ellipse, v_plot_range = 100, scale_factor=4,output_file = datapath+'/pv-original-%03i.fits' % ellipse['id'])
    subcube = SpectralCube.read(datapath+'/pv-original-%03i.fits' % ellipse['id'])
    bubble_pv_plot_v5(subcube, 38, plot_paths=True, plot_name='t1+t2/pv-original-%03i.pdf' %
                      ellipse['id'], ellipse=ellipse, reg_id=ellipse['id'], collide = ellipse['collide'],
                      reg_file='b86_J2000.reg', plot_extra_ellipses=True)

