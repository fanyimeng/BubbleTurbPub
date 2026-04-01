import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
import seaborn as sns
from num2tex import num2tex

from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy import coordinates
from astropy.table import QTable, Table, Column

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ======== KDE 画图函数（右图） ========
def kdeplot(data, kdecolor, kdeorder, kdeax, text):
    xx = np.arange(np.nanmin(data), np.nanmax(data), 0.0001)
    xx = np.log10(xx)
    kde = stats.gaussian_kde(data)
    facecolor = (*kdecolor[:3], 0.2)
    sns.histplot(data, kde=True, bins=10, color=kdecolor, edgecolor=kdecolor, log_scale=False, label=text, ax=kdeax)
    kdeax.plot(data, np.full_like(data, -0.03 * kdeorder - 0.1), '|', markeredgewidth=1, color=kdecolor)
    mymean = num2tex(np.nanmean(data), precision=2)
    mymed = num2tex(np.nanmedian(data), precision=2)
    print('$\\leftarrow \\rm mean={}$'.format(num2tex(13.6e10, precision=2)))
    return [np.power(10, xx), kde(xx)]

# ======== 椭圆误差分析函数（左图） ========
def sample_ellipse_points(N, a, b):
    t = np.linspace(0, 2 * np.pi, N)
    x = a * np.cos(t)
    y = b * np.sin(t)
    return x, y

def incline_ellipse(x, y, theta, phi):
    rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    x, y = np.dot(rotation_matrix, np.array([x, y]))
    y = y * np.cos(np.radians(theta))
    return x, y

def max_distance(x, y):
    max_dist = 0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            max_dist = max(max_dist, dist)
    return max_dist

# ======== 读取数据，开始绘图 ========
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# 左图 误差图
ax = axs[0]
N = 100
theta = 77
phis = np.linspace(0, np.pi/2, 6)
betas = np.linspace(1, 2.0, 50)

for i, phi in enumerate(phis):
    ratios = []
    for beta in betas:
        a = np.sqrt(beta)
        b = 1 / np.sqrt(beta)
        d = np.sqrt(a * b)
        x, y = sample_ellipse_points(N, a, b)
        x, y = incline_ellipse(x, y, theta, phi)
        max_dist = max_distance(x, y)
        ratio = max_dist / d / 2
        ratios.append(ratio)
    ax.plot(betas, ratios, label=f'$\\varphi$ = {(90 - np.degrees(phi)):.0f}°', color=plt.cm.viridis(i / len(phis)))

ax.set_xlabel('$a/b$')
ax.set_ylabel('$r/\\sqrt{ab}$')
ax.tick_params(axis='both', which='major', direction='in', size=10, bottom=True, top=True, left=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', size=6, bottom=True, top=True, left=True, right=True)
ax.legend(loc='upper right')
ax.text(0.08, 0.92, '(a)', transform=ax.transAxes, fontsize=10, va='top', ha='left')

# 右图 KDE
ax2 = axs[1]
df = pd.read_fwf("../code/1113.tab", header=0, infer_nrows=int(1e6))
x = abs(df['maj_as']/df['min_as'])
kdeplot(data=x, kdeorder=0, kdecolor=(0.8, 0.8, 0.5, 0.8), kdeax=ax2, text='$M_\\mathrm{Hii}$ (All)')
ax2.set_xlabel('$\\theta_{\\rm maj}/\\theta_{\\rm min}$')
ax2.set_ylabel('Count')
ax2.tick_params(axis='both', which='major', direction='in', size=10, bottom=True, top=True, left=True, right=True)
ax2.tick_params(axis='both', which='minor', direction='in', size=6, bottom=True, top=True, left=True, right=True)
ax2.axhline(y=0, lw=1, linestyle='-')
ax2.text(0.2, 0.92, '(b)', transform=ax2.transAxes, fontsize=10, va='top', ha='left')

plt.tight_layout()
plt.savefig("a_over_b_vs_r_over_sqrtab.pdf")
plt.close()
