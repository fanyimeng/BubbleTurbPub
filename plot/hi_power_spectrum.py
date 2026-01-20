from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy import coordinates
from scipy import stats
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from astropy.table import QTable, Table, Column
from matplotlib.ticker import *
import seaborn as sns
import scipy.ndimage
from matplotlib.patches import Patch
import matplotlib.colors as colors


def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    # print(nr)
    radialprofile = tbin
    return radialprofile


def convert_2d_to_1d(image):
    """
    Convert a 2D array (image) into two 1D arrays: one for the distance of each pixel to the center,
    and another for the pixel values. The distances and values are one-to-one corresponding.

    Parameters:
    image (2D numpy array): The input 2D array representing an image.

    Returns:
    tuple: A tuple containing two 1D numpy arrays. The first array is the distances, and the second array is the pixel values.
    """
    # Calculate the center of the image
    center_y, center_x = np.array(image.shape) / 2

    # Create an array of the same shape as the image containing the distance of each pixel from the center
    y, x = np.indices(image.shape)
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Flatten the 2D arrays into 1D arrays
    distance_1d = distance.flatten()
    value_1d = image.flatten()

    return distance_1d, value_1d

# # Example usage:
# # Create a dummy 2D array (image)
# dummy_image = np.random.rand(10, 10)
# distances, values = convert_2d_to_1d(dummy_image)
# distances, values


def plotProfile(ax,
                filename='',
                color='gray',
                label='',
                scale=1,
                convert_to_K=False,
                half_size=175 * 8 / 60.,  # in arcmin
                freq=1.4204057510,  # GHz
                las=1500 * 5,
                beam=60):

    hdu = fits.open(
        filename)
    data = hdu[0].data
    # data = scipy.ndimage.zoom(data, 10, order=1)
    datacenter = [int(data.shape[0] / 2), int(data.shape[1] / 2)]
    half_size = 1500
    data = data[datacenter[0] - int(half_size):datacenter[0] + int(half_size),
                datacenter[1] - int(half_size):datacenter[1] + int(half_size)]
    if convert_to_K:
        bmaj = hdu[0].header['BMAJ'] * 3600
        bmin = hdu[0].header['BMIN'] * 3600
        data = 1.222e3 * data * 1e3 / bmaj / bmin / freq**2
        print(1.222e3  * 1e3 / bmaj / bmin / freq**2)
    print(data.shape)
    data = np.nan_to_num(data)
    data_fft = np.fft.fft2(data)
    data_fft = np.fft.fftshift(data_fft)
    data_fftabs = abs(data_fft)

    distances, values = convert_2d_to_1d(data_fftabs)
    xarr = (distances) * 41.27388 * 1.2e3 / half_size
    yarr = (values) * scale
    xarr[xarr < 4.14e3 * 60 / las] = np.NaN
    yarr[xarr < 4.14e3 * 60 / las] = np.NaN

    xarr[xarr > 4.14e3 * 60 / beam] = np.NaN
    yarr[xarr > 4.14e3 * 60 / beam] = np.NaN

    # # print(xarr)
    # yarr = radial_profile(data_fftabs, mycenter) * scale
    # yarr = yarr[:int(data.shape[0] / 2)]
    # xarr = np.array(range(len(yarr)))
    # xarr = xarr /( abs(hdu[0].header['CDELT1']) / 180. * 3.14)
    # xarr = xarr/half_size
    # xarr = xarr/2/1.414
    # print(xarr)

    # ax.set_xlim([np.nanmin(xarr) * 0.1, np.nanmax(xarr) * 4])
    # ax.set_ylim([np.nanmin(yarr) * 0.1, np.nanmax(yarr) * 4])
    # ax.scatter(xarr, yarr, color=color, label=label)
    ax.hexbin(xarr, yarr, gridsize=30, cmap=color, mincnt=0, xscale='log',
              yscale='log', bins='log', edgecolors='none', norm = colors.LogNorm(vmin=1, vmax=300))
    print(yarr.shape)
    return 0


plt.rcParams['figure.dpi'] = 75.
plt.rcParams['savefig.dpi'] = 75.
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.clf()
fig = plt.figure(figsize=(6, 6))


ax = plt.subplot(1, 1, 1)


ax.set_xscale("log")
ax.xaxis.set_major_formatter(LogFormatter())
ax.xaxis.set_minor_formatter(LogFormatter())
ax.set_yscale("log")
ax.yaxis.set_major_formatter(LogFormatter())
ax.yaxis.set_minor_formatter(LogFormatter())
ax.tick_params(axis='both', which='major', direction='in', size=6,
               bottom=True, top=True, left=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', size=4,
                    bottom=True, top=True, left=True, right=True)

# Make the tick labels in scitific notations

ax.loglog()

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.2)


# plotProfile(filename='../data/all-jcomb.fits.moment.integrated.fits',
#             ax=ax,
#             color=(0.5, 0.5, 0.5, 0.0),
#             label='SDint', convert_to_K = True)

# plotProfile(filename='../data/all-jcomb.fits.moment.integrated.fits',
#             ax=ax,
#             color=(0.3, 0.3, 0.8, 0.5),
#             label='FAST', convert_to_K = True)


# ax.set_xlim([1e0, 5e4])
# ax.set_ylim([1e5, 1e9])

# ax.set_ylabel('Jy/arcsec$^2$')
ax.set_ylabel('$\\mathrm{K\,km\,s^{-1}}$')
ax.set_xlabel('$uv$ distance / $\\lambda$')

# ax.axvline(x=4.14e3,
#            lw=2.0,
#            ls='--',
#            color=(0.5, 0.5, 0.5, 1),
#            label='COMBINED BEAM 1\'')

# plotProfile(filename='../data/all-jcomb.fits.moment.integrated.fits',
#             ax=ax,
#             color='Greys',
#             label='COMBINED', scale=1, convert_to_K = True)

beam = 60
ax.axvline(x=4.14e3 * 60 / beam,
           lw=2.0,
           ls='--',
           color=(0.8, 0.3, 0.3, 0.9),
           label='100 $x$ COMBINED 1\' ~ $\\infty$')

plotProfile(filename='../data/all-jcomb.fits.moment.integrated.fits',
            ax=ax,
            color='Greys',
            beam=beam,
            label='combined', scale=100, convert_to_K=True)


las = 1265
beam = 60
ax.axvline(x=4.14e3 * 60 / beam,
           lw=2.0,
           ls='--',
           color=(0.3, 0.3, 0.8, 0.9),
           label='VLA 1\' ~ 21\'')
ax.axvline(x=4.14e3 * 60 / las,
           lw=2.0,
           ls='--',
           color=(0.3, 0.3, 0.8, 0.9))

plotProfile(filename='../data/contsub-02.fits.moment.integrated.fits',
            ax=ax,
            color='Blues',
            las=las,
            beam=beam,
            label='VLA', scale=1, convert_to_K=True)

beam = 180
ax.axvline(x=4.14e3 * 60 / beam,
           lw=2.0,
           ls='--',
           color=(0.8, 0.3, 0.3, 0.9),
           label='0.01 $x$ FAST 3\' ~ $\\infty$')

plotProfile(filename='../data/fast-rg.fits.moment.integrated.fits',
            ax=ax,
            color='Reds',
            beam=beam,
            label='FAST', scale=0.01, convert_to_K=True)


# plotProfile(filename='../data/all-jcomb.fits.moment.integrated.fits',
#             ax=ax,
#             color=(0.8, 0.3, 0.3, 0.5),
#             label='VLA', scale=1, convert_to_K = True)

# ax.axvline(x=3087,
#            lw=2.0,
#            ls='--',
#            color=(0.8, 0.3, 0.3, 0.5),
#            label='VLA BEAM')

# ax.axvline(x=45/0.21,
#            lw=2.0,
#            ls='--',
#            color=(0.8, 0.3, 0.3, 0.5),
#            label='VLA LAS')

# ax.axvline(x=175 * 8 * 1.414 / 201 / 2,
#            lw=2.0,
#            ls='--',
#            color=(0.3, 0.8, 0.3, 0.5),
#            label='FAST RES')
legend_elements = [
    Patch(facecolor='grey', label='100 $\\times$ COMBINED 1\' ~ $\\infty$'),
    Patch(facecolor='b', label='VLA 1\' ~ 21\''),
    Patch(facecolor='r', label='0.01 $\\times$ FAST 3\' ~ $\\infty$')
]

ax.legend(handles=legend_elements, loc='lower left')
plt.savefig('./hi_power_spectrum.pdf')
plt.clf()
