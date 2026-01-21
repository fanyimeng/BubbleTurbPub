import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

def read_ds9_reg_file_to_df(filename):
    ellipses = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('ellipse'):
                line = line.strip()
                parts = line.split('(')[1].split(')')[0].split(',')
                coorx, coory = float(parts[0]), float(parts[1])
                major, minor, pa = float(parts[2]), float(parts[3]), float(parts[4])
                ellipses.append({'coorx': coorx, 'coory': coory, 'major': major, 'minor': minor, 'pa': pa})
    return ellipses

def generate_synthetic_ellipses_dataframe(real_ellipse_dict, wcs):
    data = {
        'half': [],
        'id': [],
        'ra_hms': [],
        'dec_dms': [],
        'major': [],
        'minor': [],
        'pa': [],
        'v_max': [],
        'v_min': [],
        'in_b86': [],
        'sf': [],
        'v1': [],
        'v2': [],
        'v_ang': [],
        'mass': [],
        'ek': [],
        'tdyn': [],
        'c_dist': [],
        'ek_new': [],
        'sigma_ek': [],
        'n': []
    }

    for idx, ellipse in enumerate(real_ellipse_dict):
        # Convert (x, y) to (ra, dec)
        ra_dec = wcs.all_pix2world([[ellipse['coorx'], ellipse['coory']]], 1)[0]
        ra, dec = ra_dec

        # Convert to SkyCoord to format as hms and dms
        coord = SkyCoord(ra=ra, dec=dec, unit='deg')
        ra_hms = coord.ra.to_string(unit='hourangle', sep=':', precision=1)
        dec_dms = coord.dec.to_string(unit='deg', sep=':', precision=1, alwayssign=True)

        data['half'].append('north' if ellipse['coory'] > 0 else 'south')
        data['id'].append(idx + 1)
        data['ra_hms'].append(ra_hms)
        data['dec_dms'].append(dec_dms)
        data['major'].append(ellipse['major'])
        data['minor'].append(ellipse['minor'])
        data['pa'].append(ellipse['pa'])
        data['v_max'].append(np.nan)
        data['v_min'].append(np.nan)
        data['in_b86'].append(False)
        data['sf'].append(np.nan)
        data['v1'].append(np.nan)
        data['v2'].append(np.nan)
        data['v_ang'].append(np.nan)
        data['mass'].append(np.nan)
        data['ek'].append(np.nan)
        data['tdyn'].append(np.nan)
        data['c_dist'].append(np.nan)
        data['ek_new'].append(np.nan)
        data['sigma_ek'].append(np.nan)
        data['n'].append(np.nan)

    return pd.DataFrame(data)

def save_dataframe_to_fixed_width(df, filename):
    with open(filename, 'w') as file:
        df.to_string(file, index=False, justify='left')

# Main execution
fits_image_path = '../data/jcomb_vcube_scaled2_mom0.fits'
ds9_reg_file_path = '0328-final_xy.reg'
output_file_path = '0328-final_xy.tab'

# Read the FITS image to get the WCS
with fits.open(fits_image_path) as hdul:
    wcs = WCS(hdul[0].header)

# Read the DS9 region file
ellipses = read_ds9_reg_file_to_df(ds9_reg_file_path)

# Generate the DataFrame with synthetic ellipses
df = generate_synthetic_ellipses_dataframe(ellipses, wcs)

# Save the DataFrame to a fixed-width file
save_dataframe_to_fixed_width(df, output_file_path)