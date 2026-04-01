# This is the major library for Fanyi Meng's work on bubble-turbulence in M31

import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from astropy.wcs import WCS
from spectral_cube import SpectralCube
from pvextractor import Path
from astropy.coordinates import Angle
from astropy import units as u
from pvextractor import extract_pv_slice
# from matplotlib.patches import Ellipse
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from astropy.coordinates import SkyCoord
from matplotlib.colors import FuncNorm
import pandas as pd
from astropy.io import fits
from astropy.coordinates import FK5
import matplotlib.patheffects as pe
from concurrent.futures import ThreadPoolExecutor
from astropy.wcs.utils import proj_plane_pixel_scales

def read_reg_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Parse the global properties from the reg file

def calculate_relative_endpoints(center_x, center_y, angle, length):
    angle_rad = np.radians(angle)
    delta_x = np.cos(angle_rad) * length / 2
    delta_y = np.sin(angle_rad) * length / 2

    return (center_x + delta_x, center_y + delta_y), (center_x - delta_x, center_y - delta_y)

def create_pv_path(endpoints):
    """
    Create a path suitable for pvextractor.extract_pv_slice from given endpoints.

    Parameters:
    endpoints (tuple): A tuple containing two tuples, each representing an endpoint.

    Returns:
    Path: A path object suitable for use with pvextractor.extract_pv_slice.
    """
    # Extract the two endpoints
    (x1, y1), (x2, y2) = endpoints
    # print((x1, y1), (x2, y2))

    # Create a path object
    pv_path = Path([(x1, y1), (x2, y2)])

    return pv_path

def parse_global_properties(global_line):
    global_props = {}
    matches = re.findall(r'(\b\w+\b)=(?:"([^"]+)"|(\S+))', global_line)
    for match in matches:
        key = match[0]
        value = match[1] if match[1] else match[2]
        global_props[key] = value
    return global_props

def parse_region_attributes(attributes_str):
    attributes = {}
    if attributes_str:
        for attr in re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', attributes_str):
            key, value = attr.split('=')
            attributes[key] = value.strip('"')

            # Extract and sort negative velocities for the format {-number-number} including floats
            if key == 'text':
                velocity_match = re.match(r'\{-?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\}', value)
                if velocity_match:
                    velocities = sorted(
                        [-float(velocity_match.group(1)), -float(velocity_match.group(2))],
                        reverse=True)
                    attributes['velocities'] = velocities
    return attributes

def parse_ellipse_regions(reg_content):
    region_data = {
        'global': None,
        'ellipses': []
    }

    lines = reg_content.strip().split('\n')
    ellipse_pattern = re.compile(
        r'ellipse\((?P<coorx>[\d.+-]+),(?P<coory>[\d.+-]+),(?P<major>[\d.+-]+),(?P<minor>[\d.+-]+),(?P<pa>[\d.+-]+)\)'
        r'\s*#?\s*(?P<attributes>.*)'
    )
    ellipse_id = 1  # Initialize ellipse ID

    for line in lines:
        line = line.strip()
        if line.startswith('global'):
            region_data['global'] = parse_global_properties(line)
        elif line.startswith('#') or line.startswith('image'):
            continue
        else:
            match = ellipse_pattern.match(line)
            if match:
                ellipse_props = match.groupdict()

                # Ensure major axis is larger than minor axis
                major = float(ellipse_props['major'])
                minor = float(ellipse_props['minor'])
                pa = float(ellipse_props['pa'])

                if minor > major:
                    major, minor = minor, major  # Swap values
                    pa = (pa + 90) % 180  # Adjust PA

                ellipse_props['major'] = major
                ellipse_props['minor'] = minor
                ellipse_props['pa'] = pa

                ellipse_props['attributes'] = parse_region_attributes(
                    ellipse_props['attributes'])
                ellipse_props['id'] = ellipse_id  # Assign ID
                ellipse_id += 1  # Increment ID for the next ellipse
                region_data['ellipses'].append(ellipse_props)

    return region_data

def max_min_values_in_ellipse(cube, ellipse, wcs):
    """
    Calculate the maximum and minimum values of the pixels within an ellipse in the moment 0 map of a data cube.
    """
    # Create the moment 0 map by summing along the spectral axis
    moment_0_map = cube.moment(order=0).value

    # Convert ellipse from celestial to pixel coordinates
    ellipse_pixel = celestial_to_pixel(ellipse, wcs)
    # print(ellipse_pixel)

    max_value = -np.inf
    min_value = np.inf

    for y in range(moment_0_map.shape[0]):
        for x in range(moment_0_map.shape[1]):
            if is_point_in_ellipse(x, y, ellipse_pixel):
                value = moment_0_map[y, x]
                max_value = max(max_value, value)
                min_value = min(min_value, value)

    return max_value, min_value

def celestial_to_pixel(ellipse_params, wcs):
    """
    Convert ellipse parameters from celestial coordinates to pixel coordinates.
    """
    center_ra, center_dec = ellipse_params['ra'], ellipse_params['dec']
    # in arcsec
    major, minor = ellipse_params['major'], ellipse_params['minor']
    angle = ellipse_params['pa']  # in degrees

    # Convert center to pixel coordinates
    center_sky = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)
    center_pixel = wcs.world_to_pixel(center_sky)

    # Convert major and minor axes from arcsec to pixels
    # pixel_scale = np.mean(np.abs(np.diagonal(wcs.pixel_scale_matrix)))  # average pixel scale in degrees/pixel
    major_pixel = major / 5  # Convert from arcsec to pixels
    minor_pixel = minor / 5  # Convert from arcsec to pixels

    return {'center': center_pixel, 'semi_major': major_pixel * 1.2, 'semi_minor': minor_pixel * 1.2, 'angle': angle}

def is_point_in_ellipse(x, y, ellipse):
    """
    Check if a point (x, y) is inside the given ellipse.
    """
    center_x, center_y = ellipse['center']
    semi_major, semi_minor = ellipse['semi_major'], ellipse['semi_minor']
    angle = ellipse['angle']

    # Translate and rotate the point to align with the ellipse
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))
    xd = cos_angle * (x - center_x) + sin_angle * (y - center_y)
    yd = sin_angle * (x - center_x) - cos_angle * (y - center_y)

    # Check if the point is inside the ellipse
    return (xd / semi_major) ** 2 + (yd / semi_minor) ** 2 <= 1

# Function to export a subcube
def export_subcube_v2(cube, ellipse, output_file=None, scale_factor=4, v_plot_range = 40):

    # # Find the ellipse with the given ID
    # ellipse = next((e for e in region_data if e['id'] == ellipse_id), None)
    # if ellipse is None:
    #     raise ValueError(f"No ellipse found with ID {ellipse_id}")

    # Convert RA and Dec to pixel coordinates
    wcs = cube.wcs.celestial
    ra = ellipse['ra']
    dec = ellipse['dec']
    x, y = wcs.all_world2pix([[ra, dec]], 1)[0]

    # Calculate the size of the sub-cube
    # Convert from arcsec to pixel
    if ellipse['major'] < 80:
        scale_factor = 80 / ellipse['major'] * scale_factor
    major_axis_pixel = ellipse['major'] / \
        abs(wcs.pixel_scale_matrix[0, 0]) / 3600
    n = int(scale_factor * major_axis_pixel)

    # Calculate pixel coordinates for the sub-cube
    x_start, x_end = max(int(x - n), 0), min(int(x + n), cube.shape[2])
    y_start, y_end = max(int(y - n), 0), min(int(y + n), cube.shape[1])

    # print(x_start, x_end, y_start, y_end, wcs.pixel_scale_matrix[0, 0])

    # Extract the sub-cube
    # velocity_center = np.nanmean(ellipse['velocities'])
    # # print("DEBUG:", ellipse['velocities'])
    # cube = cube.spectral_slab(
    #     (velocity_center - v_plot_range) * u.km / u.s, (velocity_center + v_plot_range) * u.km / u.s)
    cube = cube.spectral_slab(
        (ellipse['velocities'][0]-v_plot_range) * u.km / u.s, (ellipse['velocities'][1]+v_plot_range) * u.km / u.s)
    sub_cube = cube[:, y_start:y_end, x_start:x_end]

    # Check if the output file is provided, then write to FITS
    if output_file:
        sub_cube.write(output_file, format='fits', overwrite=True)

    # Return the sub-cube as a SpectralCube object
    return sub_cube, ellipse['velocities']








def export_subcube_v3(cube, ellipse, output_file=None, scale_factor=4, v_plot_factor=1.5):
    """
    Extract a subcube in memory from the main cube using ellipse parameters.
    Only writes to file if output_file is provided.

    Parameters
    ----------
    cube : SpectralCube
        The full spectral cube to extract from.
    ellipse : dict
        Dictionary with 'ra', 'dec', 'major', 'velocities' (list or tuple of [vmin, vmax]).
    output_file : str or None
        If specified, write the subcube to this FITS file.
    scale_factor : float
        Spatial scaling factor for the extraction box.
    v_plot_factor : float
        Factor to expand the velocity range.

    Returns
    -------
    sub_cube : SpectralCube
        The extracted subcube (in memory).
    velocity_range : list
        The velocity range used for slicing.
    """
    wcs = cube.wcs.celestial
    ra = ellipse['ra']
    dec = ellipse['dec']
    x, y = wcs.all_world2pix([[ra, dec]], 1)[0]

    # Calculate the size of the sub-cube in pixels
    pixel_scales = proj_plane_pixel_scales(wcs)  # deg/pix
    pix_per_arcsec = 1. / (pixel_scales[0] * 3600.)

    if ellipse['major'] < 80:
        scale_factor = 80 / ellipse['major'] * scale_factor

    major_axis_pixel = ellipse['major'] * pix_per_arcsec
    n = int(scale_factor * major_axis_pixel)

    # Clamp spatial bounds
    x_start = max(int(x - n), 0)
    x_end = min(int(x + n), cube.shape[2])
    y_start = max(int(y - n), 0)
    y_end = min(int(y + n), cube.shape[1])

    # Sort and expand velocity range by factor
    vmin, vmax = sorted(ellipse['velocities'])
    v_center = 0.5 * (vmin + vmax)
    v_half_width = 0.5 * (vmax - vmin)
    v_expanded = v_half_width * v_plot_factor
    slab = cube.spectral_slab(
        (v_center - v_expanded) * u.km/u.s,
        (v_center + v_expanded) * u.km/u.s
    )

    # Extract subcube spatially
    sub_cube = slab[:, y_start:y_end, x_start:x_end]

    if output_file:
        sub_cube.write(output_file, format='fits', overwrite=True)

    return sub_cube, [vmin, vmax]








def ellipse_line_intersection_improved(centerx, centery, major, minor, pa, pa_seg):
    from scipy.optimize import fsolve
    # Convert angles from degrees to radians
    pa = np.radians(pa)
    pa_seg = np.radians(pa_seg)

    # Define the equations to solve
    def equations(p):
        x, y = p
        eq1 = ((x - centerx) * np.cos(pa) + (y - centery) * np.sin(pa)) ** 2 / major**2 + \
              ((x - centerx) * np.sin(pa) - (y - centery) * np.cos(pa)) ** 2 / minor**2 - 1
        eq2 = y - (centery + (x - centerx) * np.tan(pa_seg))
        return (eq1, eq2)

    # Initial guess for the first solution
    initial_guess1 = (centerx + major * np.cos(pa_seg), centery + major * np.sin(pa_seg))

    # Solve the equations for the first intersection
    intersect1 = fsolve(equations, initial_guess1)

    # Initial guess for the second solution, using a point on the opposite side of the ellipse
    initial_guess2 = (centerx - major * np.cos(pa_seg), centery - major * np.sin(pa_seg))

    # Solve the equations for the second intersection
    intersect2 = fsolve(equations, initial_guess2)

    return np.array(intersect1), np.array(intersect2)
























def bubble_pv_plot_v4(cube, theta, plot_paths=False, plot_name='test.pdf', ellipse={}, reg_id='', collide = False):
    wcs = cube.wcs.celestial
    center_x, center_y = cube.shape[2] / 2, cube.shape[1] / 2
    length = int(np.min([cube.shape[2], cube.shape[1]])*0.8)

    endpoints1 = calculate_relative_endpoints(
        center_x, center_y, theta, length)
    endpoints2 = calculate_relative_endpoints(
        center_x, center_y, theta + 90, length)

    central_slice = cube[int(cube.shape[0] / 2) -
                         1: int(cube.shape[0] / 2) + 1, :, :]

    velocity_range_slice = cube.spectral_slab(np.nanmin(ellipse['velocities']) * u.km / u.s, np.nanmax(ellipse['velocities']) * u.km / u.s) 

    max_val, min_val = max_min_values_in_ellipse(central_slice, ellipse, wcs)

    ell_major = ellipse['major'] / 5.0 * 2.0
    ell_minor = ellipse['minor'] / 5.0 * 2.0
    ell_pa = ellipse['pa'] - 90.

    my_ellipse = patches.Ellipse(xy=(center_x, center_y), width=ell_minor, height=ell_major, 
                    angle = ell_pa, edgecolor='y', fc='None', lw=2, ls = '--')
    
    if plot_paths:
        fig = plt.figure(figsize=(16, 8))
        fig.subplots_adjust(left=0.05, right=0.97,
                            bottom=0.05, top=0.97)

        ax = plt.subplot(2, 4, 1, projection=wcs)

        cube_moment = central_slice.moment(order=0).value
        ax.imshow(cube_moment, cmap=plt.cm.gray,
                  interpolation='nearest',
                  origin='lower', aspect='equal',
                  norm=mcolors.PowerNorm(gamma=1.0,
                                         vmax=max_val, vmin=0))
        ax.set_xlabel(f"Right Ascension [{cube.wcs.wcs.radesys}]")
        ax.set_ylabel(f"Declination [{cube.wcs.wcs.radesys}]")
        ax0 = ax.coords[0]
        ax0.tick_params(direction='in')
        ax1 = ax.coords[1]
        ax1.tick_params(direction='in')

        ax.add_patch(my_ellipse)

        def plot_path(endpoints, color):
            x_coords, y_coords = zip(*endpoints)
            ax.plot(x_coords, y_coords, color=color, ls='--')

        shadow_dist = 0.005

        if collide:
            plt.text(0.1+shadow_dist, 0.9-shadow_dist, reg_id,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes, size=18, color='w', fontweight = 'bold')
        else:
            plt.text(0.1+shadow_dist, 0.9-shadow_dist, reg_id,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes, size=18, color='k', fontweight = 'bold')
        shadow_dist = 0
        if collide:
            plt.text(0.1+shadow_dist, 0.9-shadow_dist, reg_id,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes, size=18, color='r', fontweight = 'bold')
        else:
            plt.text(0.1+shadow_dist, 0.9-shadow_dist, reg_id,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes, size=18, color='w', fontweight = 'bold')
        shadow_dist = 0.005
        path_list = []
        end_point_list = []
        ang_list = []
        for angle in range(6):
            angle_deg = angle * 30
            ang_list.append(theta + angle_deg)
            end_point_list.append(
                calculate_relative_endpoints(
                    center_x, center_y, theta + angle_deg, length)
            )
            path_list.append(
                create_pv_path(
                    calculate_relative_endpoints(
                        center_x, center_y, theta + angle_deg, length)
                )
            )



        ax = plt.subplot(2, 4, 5, projection=wcs)
        cube_vrange_moment = velocity_range_slice.moment(order=0).value
        max_val_vrange, min_val_vrange = max_min_values_in_ellipse(velocity_range_slice, ellipse, wcs)
        ax.imshow(cube_vrange_moment, cmap=plt.cm.gray,
                  interpolation='nearest',
                  origin='lower', aspect='equal',
                  vmin=min_val_vrange, vmax=max_val_vrange)
        ax.set_xlabel(f"Right Ascension [{cube.wcs.wcs.radesys}]")
        ax.set_ylabel(f"Declination [{cube.wcs.wcs.radesys}]")
        ax0 = ax.coords[0]
        ax0.tick_params(direction='in')
        ax1 = ax.coords[1]
        ax1.tick_params(direction='in')

        for i, end_point in enumerate(end_point_list):
            plot_path(end_point, color=(0.5,0.5, 0.8, 0.9))
            plot_pos = np.array(end_point[0]) * \
                0.9 + np.array(end_point[1]) * 0.1
            ax.text(*plot_pos + [0.2, -0.2], str(ang_list[i]) + '$^\\circ$', size=14, color=(0.0,0.0, 0.0, 1))
            ax.text(*plot_pos, str(ang_list[i]) + '$^\\circ$', size=14, color=(0.6,0.6, 0.8, 1))
        plot_path(endpoints2, color=(0.9, 0.1, 0.1, 1.0))


        ell_major = ellipse['major'] / 5.0 * 2.0
        ell_minor = ellipse['minor'] / 5.0 * 2.0
        ell_pa = ellipse['pa'] - 90.

        from matplotlib.patches import Ellipse
        my_ellipse = Ellipse(xy=(center_x, center_y), width=ell_minor, height=ell_major, 
                        angle = ell_pa, edgecolor='y', fc='None', lw=2, ls = '--')
        ax.add_patch(my_ellipse)

        for i, path in enumerate(path_list):
            pv_diagram = extract_pv_slice(cube, path)
            aspect_ratio = float(pv_diagram.shape[1] / pv_diagram.shape[0])
            ww = WCS(pv_diagram.header)

            ax = plt.subplot(2, 4, 2 + i + int(i > 2), projection=ww)
            max_pvval = np.nanmax(pv_diagram.data)
            min_pvval = np.nanmin(pv_diagram.data)

            norm = FuncNorm((np.arcsinh, np.sinh))
            # Plot the PV diagram
            ax.imshow(pv_diagram.data, aspect=aspect_ratio,
                      origin='lower',
                      norm=norm, cmap=plt.cm.RdYlBu_r)
                      # norm=mcolors.PowerNorm(
                      #     gamma=1.5, 
                      #     vmax = max_pvval-(max_pvval - min_pvval)*0.2, 
                      #     vmin = min_pvval+(max_pvval - min_pvval)*0.2), 
                      #     cmap=plt.cm.RdYlBu_r)  # Assuming pv_diagram is a 2D array
            ax0 = ax.coords[0]
            ax0.set_format_unit(u.arcmin)
            ax0.tick_params(direction='in')
            ax0.display_minor_ticks(True)
            ax1 = ax.coords[1]
            ax1.set_format_unit(u.km / u.s)
            ax1.tick_params(direction='in')
            ax1.display_minor_ticks(True)
            ax.invert_yaxis()

            plt.text(0.1, 0.9, str(ang_list[i]) + '$^\\circ$',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax.transAxes, size=14)


            ax.set_ylabel("Velocity [km/s]")
            ax.set_xlabel("Offset [arcmin]")

            if abs(ang_list[i] - ellipse['v_ang']) < 1.:
                p_center = np.nanmean(ax.get_xlim())
                v_low = (ellipse['v1'] * 1e3 - cube.header['CRVAL3'])/cube.header['CDELT3'] + cube.header['CRPIX3']
                v_high = (ellipse['v2'] * 1e3 - cube.header['CRVAL3'])/cube.header['CDELT3'] + cube.header['CRPIX3']
                ax.plot([p_center, p_center],
                    [v_low, v_high],
                    color = 'gray', ls = '--', marker = '_', lw = 2, markersize = 15)

            cdelt3 = cube.header['CDELT3']*1e-3
            deltav = ellipse['velocities'][0] - ellipse['velocities'][1]
            deltav_pix = abs(deltav / cdelt3)
            v_low = np.nanmean(ax.get_ylim()) - deltav_pix/2.
            v_high = np.nanmean(ax.get_ylim()) + deltav_pix/2.

            intersect1, intersect2 = ellipse_line_intersection_improved(center_x, center_y, ell_major / 2., ell_minor / 2., ell_pa, ang_list[i]-90.)
            interdist = np.sqrt(np.sum((intersect1-intersect2)**2))

            p_low = np.nanmean(ax.get_xlim()) - interdist/2.
            p_high = np.nanmean(ax.get_xlim()) + interdist/2.

            ax.plot([p_low, p_low], [v_low, v_high], color = 'gray', 
                ls = '--', marker = '_', lw = 2, markersize = 15,
                path_effects=[pe.Stroke(linewidth=5, foreground='black'),
                              pe.Normal()])
            ax.plot([p_high, p_high], [v_low, v_high], color = 'gray', 
                ls = '--', marker = '_', lw = 2, markersize = 15,
                path_effects=[pe.Stroke(linewidth=5, foreground='black'),
                              pe.Normal()])



        def sum_pixels_in_ellipse(array, center_x, center_y, major, minor, pa):
            """
            Sum the pixels within an ellipse in a 2D numpy array.

            :param array: 2D numpy array
            :param center_x: X-coordinate of the ellipse's center
            :param center_y: Y-coordinate of the ellipse's center
            :param major: Major axis length of the ellipse
            :param minor: Minor axis length of the ellipse
            :param pa: Position angle of the ellipse in radians, measured from the x-axis
            :return: Sum of the pixel values inside the ellipse
            """
            rows, cols = array.shape
            y_indices, x_indices = np.ogrid[:rows, :cols]
            cos_pa = np.cos(pa)
            sin_pa = np.sin(pa)
            
            # Transform coordinates
            x = x_indices - center_x
            y = y_indices - center_y
            x_rot = cos_pa * x + sin_pa * y
            y_rot = -sin_pa * x + cos_pa * y

            # Create ellipse mask
            ellipse_mask = ((x_rot**2 / major**2) + (y_rot**2 / minor**2)) <= 1

            # Sum the pixels inside the ellipse
            return np.sum(array[ellipse_mask])

        mom0_cube = np.nansum(velocity_range_slice.unmasked_data[:,:,:].value, axis = 0)

        sum_mom0 = sum_pixels_in_ellipse(mom0_cube, center_x, center_y, ell_major / 1., ell_minor / 1., ell_pa + 90.)


        plt.savefig('%s' % (plot_name))


        sum_mom0_kkms = sum_mom0 * 0.88299 * (5/60)**2 
        sum_mom0_kkms = sum_mom0 * 1.222e6 / 1.4**2 / 60**2
        sum_g = 1.8e18 * sum_mom0_kkms * 1.67e-24
        sum_g = sum_g * (5/60*216*3e18)**2
        sum_smass = sum_g/2e33
        ek = sum_g * ((ellipse['v1'] - ellipse['v2']) * 1e5)**2 / 8
        plt.clf()
        plt.close()

        return sum_smass, ek




















import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Ellipse
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.wcs.utils import proj_plane_pixel_scales
import re

def read_ds9_ellipse_regions(reg_path):
    """
    读取 DS9 的椭圆 region 文件，返回一个列表，每个元素为字典，包含：
    'ra' (字符串, 如 "0:38:09.9"), 'dec' (字符串, 如 "+39:51:17.7"),
    'major' (弧秒), 'minor' (弧秒), 'pa' (度), 'label' (文本标注)
    """
    regions = []
    with open(reg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("ellipse"):
                match = re.match(r'ellipse\(([^,]+),([^,]+),([^"]+)"\s*,\s*([^"]+)"\s*,\s*([^)]+)\)\s*#\s*text={(.*)}', line)
                if match:
                    ra, dec, major, minor, pa, label = match.groups()
                    regions.append({
                        'ra': ra.strip(),
                        'dec': dec.strip(),
                        'major': float(major),
                        'minor': float(minor),
                        'pa': float(pa),
                        'label': label.strip()
                    })
    return regions

def bubble_pv_plot_v5(cube, theta, plot_paths=False, plot_name='test.pdf', ellipse={}, 
                      reg_id='', collide=False, reg_file=None, plot_extra_ellipses=False):
    """
    绘制 HI bubble 的 PV 图和相关图像，功能包括：
      1. 利用 cube 和给定 theta 绘制 PV 图和 moment-0 图
      2. 在左侧两个 RA/Dec 图（子图 1 和 5）中叠加绘制椭圆，
         其中主要椭圆由参数 ellipse 给出；另外，可选地叠加来自 reg_file 的 DS9 椭圆（用红色显示，并标注文字）。
      
    参数：
      cube            : 3D FITS 数据 cube，含 WCS 信息
      theta           : 主方向角（度）
      plot_paths      : 是否绘制路径（布尔）
      plot_name       : 输出图像文件名
      ellipse         : 字典，包含 'major','minor','pa','velocities','v_ang','v1','v2' 等参数
      reg_id          : 主 region 的标识文字
      collide         : 是否碰撞（影响文字颜色）
      reg_file        : 可选，DS9 region 文件路径（椭圆）
      plot_extra_ellipses : 布尔，是否在左侧图中叠加绘制 reg_file 中的椭圆
      
    返回：
      sum_smass, ek  (质量、动能)
    """
    # 主图 WCS（celestial 部分）
    wcs = cube.wcs.celestial
    center_x, center_y = cube.shape[2] / 2, cube.shape[1] / 2
    length = int(np.min([cube.shape[2], cube.shape[1]]) * 0.8)

    endpoints1 = calculate_relative_endpoints(center_x, center_y, theta, length)
    endpoints2 = calculate_relative_endpoints(center_x, center_y, theta + 90, length)

    central_slice = cube[int(cube.shape[0] / 2) - 1: int(cube.shape[0] / 2) + 1, :, :]

    velocity_range_slice = cube.spectral_slab(np.nanmin(ellipse['velocities']) * u.km / u.s, 
                                              np.nanmax(ellipse['velocities']) * u.km / u.s)

    max_val, min_val = max_min_values_in_ellipse(central_slice, ellipse, wcs)

    ell_major = ellipse['major'] / 5.0 * 2.0
    ell_minor = ellipse['minor'] / 5.0 * 2.0
    ell_pa = ellipse['pa'] - 90.

    my_ellipse = Ellipse(xy=(center_x, center_y), width=ell_minor, height=ell_major, 
                         angle=ell_pa, edgecolor='y', fc='None', lw=2, ls='--')

    if plot_paths:
        fig = plt.figure(figsize=(16, 8))
        fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97)

        # --- 子图 1：moment-0 图 ---
        ax1 = plt.subplot(2, 4, 1, projection=wcs)
        cube_moment = central_slice.moment(order=0).value
        # norm = FuncNorm((np.arcsinh, np.sinh))
        ax1.imshow(cube_moment, cmap=plt.cm.gray, interpolation='nearest',
                   origin='lower', aspect='equal',
                   norm=mcolors.PowerNorm(gamma=1.0, vmax=max_val, vmin=0))
        ax1.set_xlabel(f"Right Ascension [{cube.wcs.wcs.radesys}]")
        ax1.set_ylabel(f"Declination [{cube.wcs.wcs.radesys}]")
        ax1.coords[0].tick_params(direction='in')
        ax1.coords[1].tick_params(direction='in')
        ax1.add_patch(my_ellipse)

        # 原有文字标注
        shadow_dist = 0.00
        if not collide:
            plt.text(0.1 + shadow_dist, 0.9 - shadow_dist, reg_id,
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax1.transAxes, size=18, color='k', 
                path_effects=[pe.Stroke(linewidth=3, foreground='white'),
                              pe.Normal()])
        shadow_dist = 0
        if collide:
        #     plt.text(0.1 + shadow_dist, 0.9 - shadow_dist, reg_id,
        #              horizontalalignment='center', verticalalignment='center',
        #              transform=ax1.transAxes, size=18, color='w', fontweight='bold')
        # else:
            plt.text(0.1 + shadow_dist, 0.9 - shadow_dist, reg_id,
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax1.transAxes, size=18, color='r', 
                path_effects=[pe.Stroke(linewidth=3, foreground='white'),
                              pe.Normal()])

        # 如果设置了叠加额外椭圆，则在子图 1 上绘制 reg_file 中的椭圆（红色）
        if plot_extra_ellipses and reg_file is not None:
            regions = read_ds9_ellipse_regions(reg_file)
            # 获取像素尺度（arcsec/pixel）
            pix_scales = proj_plane_pixel_scales(cube.wcs.celestial)  # in deg/pix
            pix_scale_arcsec = pix_scales[0] * 3600  # 假设 x,y 同尺度
            for reg in regions:
                # 将 DS9 region 中的 (ra, dec) 转换到像素坐标
                skycoord = SkyCoord(ra=reg['ra'], dec=reg['dec'], unit=(u.hourangle, u.deg), frame=FK5(equinox='J2000'))
                pix_coord = cube.wcs.celestial.wcs_world2pix([[skycoord.ra.deg, skycoord.dec.deg]], 1)[0]
                # 转换椭圆尺寸：major/minor (arcsec) -> pixel
                width_pix = reg['major'] / pix_scale_arcsec
                height_pix = reg['minor'] / pix_scale_arcsec
                angle_patch = reg['pa'] - 90.0  # 转换 DS9 PA 到 patch 所需角度
                extra_ellipse = Ellipse(xy=(pix_coord[0], pix_coord[1]), width=width_pix,
                                        height=height_pix, angle=angle_patch,
                                        edgecolor='red', fc='None', lw=2)
                ax1.add_patch(extra_ellipse)
                # 标注文字：将文字放在椭圆中心附近
                ax1.text(pix_coord[0], pix_coord[1], reg['label'], color='red', fontsize=10,
                         horizontalalignment='center', verticalalignment='center')

        # --- 子图 5：速度范围 moment-0 图 ---
        ax5 = plt.subplot(2, 4, 5, projection=wcs)
        cube_vrange_moment = velocity_range_slice.moment(order=0).value
        max_val_vrange, min_val_vrange = max_min_values_in_ellipse(velocity_range_slice, ellipse, wcs)
        ax5.imshow(cube_vrange_moment, cmap=plt.cm.gray, interpolation='nearest',
                   origin='lower', aspect='equal', vmin=min_val_vrange, vmax=max_val_vrange)
        ax5.set_xlabel(f"Right Ascension [{cube.wcs.wcs.radesys}]")
        ax5.set_ylabel(f"Declination [{cube.wcs.wcs.radesys}]")
        ax5.coords[0].tick_params(direction='in')
        ax5.coords[1].tick_params(direction='in')
        my_ellipse = Ellipse(xy=(center_x, center_y), width=ell_minor, height=ell_major, 
                         angle=ell_pa, edgecolor='y',alpha = 0.6, fc='None', lw=1, ls='--')
        ax5.add_patch(my_ellipse)
        # 同样在子图 5 叠加额外椭圆（红色）
        if plot_extra_ellipses and reg_file is not None:
            regions = read_ds9_ellipse_regions(reg_file)
            for reg in regions:
                skycoord = SkyCoord(ra=reg['ra'], dec=reg['dec'], unit=(u.hourangle, u.deg), frame=FK5(equinox='J2000'))
                pix_coord = cube.wcs.celestial.wcs_world2pix([[skycoord.ra.deg, skycoord.dec.deg]], 1)[0]
                width_pix = reg['major'] / pix_scale_arcsec
                height_pix = reg['minor'] / pix_scale_arcsec
                angle_patch = reg['pa'] - 90.0
                extra_ellipse = Ellipse(xy=(pix_coord[0], pix_coord[1]), width=width_pix,
                                        height=height_pix, angle=angle_patch,
                                        edgecolor='red', fc='None', lw=2)
                ax5.add_patch(extra_ellipse)
                ax5.text(pix_coord[0], pix_coord[1], reg['label'], color='red', fontsize=10,
                         horizontalalignment='center', verticalalignment='center')

        # 以下代码保持不变：绘制 PV 图、路径、其他子图等……
        path_list = []
        end_point_list = []
        ang_list = []
        for angle in range(6):
            angle_deg = angle * 30
            ang_list.append(theta + angle_deg)
            end_point_list.append(calculate_relative_endpoints(center_x, center_y, theta + angle_deg, length))
            path_list.append(create_pv_path(calculate_relative_endpoints(center_x, center_y, theta + angle_deg, length)))

        for i, path in enumerate(path_list):
            pv_diagram = extract_pv_slice(cube, path)
            aspect_ratio = float(pv_diagram.shape[1] / pv_diagram.shape[0])
            ww = WCS(pv_diagram.header)
            ax = plt.subplot(2, 4, 2 + i + int(i > 2), projection=ww)
            max_pvval = np.nanmax(pv_diagram.data)
            min_pvval = np.nanmin(pv_diagram.data)
            norm = FuncNorm((np.arcsinh, np.sinh))
            ax.imshow(pv_diagram.data, aspect=aspect_ratio, origin='lower',
                      norm=norm, cmap=plt.cm.RdYlBu_r)
            ax.coords[0].set_format_unit(u.arcmin)
            ax.coords[0].tick_params(direction='in')
            ax.coords[0].display_minor_ticks(True)
            ax.coords[1].set_format_unit(u.km / u.s)
            ax.coords[1].tick_params(direction='in')
            ax.coords[1].display_minor_ticks(True)
            ax.invert_yaxis()
            plt.text(0.1, 0.9, str(ang_list[i]) + '$^\\circ$', horizontalalignment='center',
                     verticalalignment='center', transform=ax.transAxes, size=14)
            ax.set_ylabel("Velocity [km/s]")
            ax.set_xlabel("Offset [arcmin]")
            mycolor = (0.1,0.1,0.1,1)
            mycolor_red = (0.1,0.1,0.1,1)
            if abs(ang_list[i] - ellipse['v_ang']) < 1.:
                p_center = np.nanmean(ax.get_xlim())
                v_low = (ellipse['v1'] * 1e3 - cube.header['CRVAL3']) / cube.header['CDELT3'] + cube.header['CRPIX3'] -1 
                v_high = (ellipse['v2'] * 1e3 - cube.header['CRVAL3']) / cube.header['CDELT3'] + cube.header['CRPIX3'] -1
                ax.plot([p_center, p_center], [v_low, v_high], 
                    color=mycolor_red , ls='--', marker='$-$', lw=1.5, markersize=15,
                    path_effects=[pe.Stroke(linewidth=3, foreground='white'),
                              pe.Normal()])
            cdelt3 = cube.header['CDELT3'] * 1e-3
            deltav = ellipse['velocities'][0] - ellipse['velocities'][1]
            deltav_pix = abs(deltav / cdelt3)
            v_low = np.nanmean(ax.get_ylim()) - deltav_pix / 2.
            v_high = np.nanmean(ax.get_ylim()) + deltav_pix / 2.
            intersect1, intersect2 = ellipse_line_intersection_improved(center_x, center_y, ell_major / 2., ell_minor / 2., ell_pa, ang_list[i] - 90.)
            interdist = np.sqrt(np.sum((intersect1 - intersect2) ** 2))
            p_low = np.nanmean(ax.get_xlim()) - interdist / 2.
            p_high = np.nanmean(ax.get_xlim()) + interdist / 2.
            ax.plot([p_low, p_low], [v_low, v_high], 
                color=mycolor, ls='--', marker='$-$', lw=1.5, markersize=15,
                path_effects=[pe.Stroke(linewidth=3, foreground='white'),
                              pe.Normal()])
            ax.plot([p_high, p_high], [v_low, v_high], 
                color=mycolor, ls='--', marker='$-$', lw=1.5, markersize=15,
                path_effects=[pe.Stroke(linewidth=3, foreground='white'),
                              pe.Normal()])

        def sum_pixels_in_ellipse(array, center_x, center_y, major, minor, pa):
            rows, cols = array.shape
            y_indices, x_indices = np.ogrid[:rows, :cols]
            cos_pa = np.cos(pa)
            sin_pa = np.sin(pa)
            x = x_indices - center_x
            y = y_indices - center_y
            x_rot = cos_pa * x + sin_pa * y
            y_rot = -sin_pa * x + cos_pa * y
            ellipse_mask = ((x_rot ** 2 / major ** 2) + (y_rot ** 2 / minor ** 2)) <= 1
            return np.sum(array[ellipse_mask])

        mom0_cube = np.nansum(velocity_range_slice.unmasked_data[:, :, :].value, axis=0)
        sum_mom0 = sum_pixels_in_ellipse(mom0_cube, center_x, center_y, ell_major / 1., ell_minor / 1., ell_pa + 90.)
        
        fig = plt.gcf()  # 获取当前有效的 figure（get current figure）
        plt.savefig(plot_name)
        plt.close(fig)
        # sum_mom0_kkms = sum_mom0 * 0.88299 * (5 / 60) ** 2 
        # sum_mom0_kkms = sum_mom0 * 1.222e6 / 1.4 ** 2 / 60 ** 2
        # sum_g = 1.8e18 * sum_mom0_kkms * 1.67e-24
        # sum_g = sum_g * (5 / 60 * 216 * 3e18) ** 2
        # sum_smass = sum_g / 2e33
        # ek = sum_g * ((ellipse['v1'] - ellipse['v2']) * 1e5) ** 2 / 8
        # plt.clf()
        # plt.close()
        # return sum_smass, ek
        return 0













def bubble_pv_plot_v6(cube, theta, plot_paths=False, plot_name='test.pdf', ellipse={}, 
                      reg_id='', collide=False, reg_file=None, plot_extra_ellipses=False):
    """
    绘制 HI bubble 的 PV 图和相关图像，功能包括：
      1. 利用 cube 和给定 theta 绘制 PV 图和 moment-0 图
      2. 在左侧两个 RA/Dec 图（子图 1 和 5）中叠加绘制椭圆，
         其中主要椭圆由参数 ellipse 给出；另外，可选地叠加来自 reg_file 的 DS9 椭圆（用红色显示，并标注文字）。
      
    参数：
      cube            : 3D FITS 数据 cube，含 WCS 信息
      theta           : 主方向角（度）
      plot_paths      : 是否绘制路径（布尔）
      plot_name       : 输出图像文件名
      ellipse         : 字典，包含 'major','minor','pa','velocities','v_ang','v1','v2' 等参数
                        其中 'major' 与 'minor' 都是直径，单位 arcsec；'pa' 单位度；'velocities' km/s
      reg_id          : 主 region 的标识文字
      collide         : 是否碰撞（影响文字颜色）
      reg_file        : 可选，DS9 region 文件路径（椭圆）
      plot_extra_ellipses : 布尔，是否在左侧图中叠加绘制 reg_file 中的椭圆
      
    返回：
      sum_smass, ek  (质量、动能)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Ellipse
    import matplotlib.patheffects as pe
    from astropy import units as u
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    from astropy.coordinates import SkyCoord, FK5

    # ============ 1. 基础信息 & 单位转换 ============
    wcs_2d = cube.wcs.celestial
    center_x = cube.shape[2] / 2.0
    center_y = cube.shape[1] / 2.0

    # 像素尺度（deg/pix -> arcsec/pix）
    pix_scales = proj_plane_pixel_scales(wcs_2d)
    pix_scale_arcsec = pix_scales[0] * 3600.0

    # 把 ellipse 的 major/minor (arcsec) 换成像素
    ell_major_pix = ellipse['major'] / pix_scale_arcsec
    ell_minor_pix = ellipse['minor'] / pix_scale_arcsec
    ell_pa = ellipse['pa'] - 90.0  # DS9 与 matplotlib Patch 的角度差

    # 想要在图上显示“刚好包住椭圆”的区域，可再加一点系数 margin_factor
    margin_factor = 2.5  # 若想多留空白可改大些，比如 1.2, 2.0, etc.
    half_width_pix  = 0.5 * margin_factor * ell_minor_pix
    half_height_pix = 0.5 * margin_factor * ell_major_pix

    # 计算像素坐标范围
    x_min_pix = center_x - half_width_pix
    x_max_pix = center_x + half_width_pix
    y_min_pix = center_y - half_height_pix
    y_max_pix = center_y + half_height_pix

    # 速度范围
    vel_min = np.nanmin(ellipse['velocities'])
    vel_max = np.nanmax(ellipse['velocities'])

    # ============ 2. 提取两个 moment-0 ============

    # (1) 中心附近的 moment-0
    central_slice = cube[int(cube.shape[0] / 2) - 1 : int(cube.shape[0] / 2) + 1, :, :]
    cube_moment = central_slice.moment(order=0).value

    # (2) 速度范围内的 moment-0
    velocity_range_slice = cube.spectral_slab(vel_min * u.km / u.s, vel_max * u.km / u.s)
    cube_vrange_moment = velocity_range_slice.moment(order=0).value

    # 用 ellipse 区域统计 min/max，避免颜色失真
    max_val, min_val = max_min_values_in_ellipse(central_slice, ellipse, wcs_2d)
    max_val_vrange, min_val_vrange = max_min_values_in_ellipse(velocity_range_slice, ellipse, wcs_2d)

    # ============ 3. 开始画图 ============

    if plot_paths:
        fig = plt.figure(figsize=(16, 8))
        fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97)

        # ---------- 子图 1：moment-0 (中心通道) ----------
        ax1 = plt.subplot(2, 4, 1, projection=wcs_2d)
        im1 = ax1.imshow(cube_moment,
                         cmap=plt.cm.gray,
                         interpolation='nearest',
                         origin='lower',
                         aspect='auto',  # 这里改成 'auto'，避免强行画成方形
                         norm=mcolors.PowerNorm(gamma=1.0, vmax=max_val, vmin=0))

        # 用像素范围限制可视区域
        ax1.set_xlim(x_min_pix, x_max_pix)
        ax1.set_ylim(y_min_pix, y_max_pix)

        ax1.set_xlabel(f"Right Ascension [{wcs_2d.wcs.radesys}]")
        ax1.set_ylabel(f"Declination [{wcs_2d.wcs.radesys}]")
        ax1.coords[0].tick_params(direction='in')
        ax1.coords[1].tick_params(direction='in')

        # 在图上加个主椭圆
        my_ellipse = Ellipse(xy=(center_x, center_y),
                             width=ell_minor_pix,
                             height=ell_major_pix,
                             angle=ell_pa,
                             edgecolor='y',
                             fc='None', lw=2, ls='--')
        ax1.add_patch(my_ellipse)

        # 标注文字
        shadow_dist = 0.0
        if not collide:
            ax1.text(0.1 + shadow_dist, 0.9 - shadow_dist, reg_id,
                     ha='center', va='center', transform=ax1.transAxes,
                     size=18, color='k',
                     path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])
        else:
            ax1.text(0.1 + shadow_dist, 0.9 - shadow_dist, reg_id,
                     ha='center', va='center', transform=ax1.transAxes,
                     size=18, color='r',
                     path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

        # 如果要叠加额外椭圆
        if plot_extra_ellipses and reg_file is not None:
            regions = read_ds9_ellipse_regions(reg_file)
            for reg in regions:
                skycoord = SkyCoord(ra=reg['ra'], dec=reg['dec'],
                                    unit=(u.hourangle, u.deg), frame=FK5(equinox='J2000'))
                pix_coord = wcs_2d.wcs_world2pix([[skycoord.ra.deg, skycoord.dec.deg]], 1)[0]
                width_pix = reg['major'] / pix_scale_arcsec
                height_pix = reg['minor'] / pix_scale_arcsec
                angle_patch = reg['pa'] - 90.0
                extra_ellipse = Ellipse(xy=(pix_coord[0], pix_coord[1]),
                                        width=width_pix, height=height_pix,
                                        angle=angle_patch,
                                        edgecolor='red', fc='None', lw=2)
                ax1.add_patch(extra_ellipse)
                ax1.text(pix_coord[0], pix_coord[1], reg['label'],
                         color='red', fontsize=10, ha='center', va='center')

        # ---------- 子图 5：moment-0 (速度范围) ----------
        ax5 = plt.subplot(2, 4, 5, projection=wcs_2d)
        im5 = ax5.imshow(cube_vrange_moment,
                         cmap=plt.cm.gray,
                         interpolation='nearest',
                         origin='lower',
                         aspect='auto',  # 同样改成 'auto'
                         vmin=min_val_vrange, vmax=max_val_vrange)

        ax5.set_xlim(x_min_pix, x_max_pix)
        ax5.set_ylim(y_min_pix, y_max_pix)

        ax5.set_xlabel(f"Right Ascension [{wcs_2d.wcs.radesys}]")
        ax5.set_ylabel(f"Declination [{wcs_2d.wcs.radesys}]")
        ax5.coords[0].tick_params(direction='in')
        ax5.coords[1].tick_params(direction='in')

        my_ellipse_v = Ellipse(xy=(center_x, center_y),
                               width=ell_minor_pix, height=ell_major_pix,
                               angle=ell_pa,
                               edgecolor='y', alpha=0.6, fc='None', lw=1, ls='--')
        ax5.add_patch(my_ellipse_v)

        if plot_extra_ellipses and reg_file is not None:
            regions = read_ds9_ellipse_regions(reg_file)
            for reg in regions:
                skycoord = SkyCoord(ra=reg['ra'], dec=reg['dec'],
                                    unit=(u.hourangle, u.deg), frame=FK5(equinox='J2000'))
                pix_coord = wcs_2d.wcs_world2pix([[skycoord.ra.deg, skycoord.dec.deg]], 1)[0]
                width_pix = reg['major'] / pix_scale_arcsec
                height_pix = reg['minor'] / pix_scale_arcsec
                angle_patch = reg['pa'] - 90.0
                extra_ellipse = Ellipse(xy=(pix_coord[0], pix_coord[1]),
                                        width=width_pix, height=height_pix,
                                        angle=angle_patch,
                                        edgecolor='red', fc='None', lw=2)
                ax5.add_patch(extra_ellipse)
                ax5.text(pix_coord[0], pix_coord[1], reg['label'],
                         color='red', fontsize=10, ha='center', va='center')

        # ============ 4. PV 图 x 轴范围修正 ============

        # 如果 ellipse['minor'] 是直径(arcsec)，希望 PV x 轴正好 ±(minor/2)
        offset_radius_arcmin = (ellipse['minor'] / 2.0) / 60.0

        # 下面只是为了生成 6 条切片 path（从theta到theta+150°每30°），你可以按自己需求改
        path_list = []
        ang_list = []
        # 这里 length 先随便取一个够长的像素数，或者你也可与 minor_pix/major_pix 对应
        length = int(max(x_max_pix - x_min_pix, y_max_pix - y_min_pix))
        for angle_step in range(6):
            angle_deg = angle_step * 30
            ang_list.append(theta + angle_deg)
            endpoints = calculate_relative_endpoints(center_x, center_y, theta + angle_deg, length)
            path_list.append(create_pv_path(endpoints))

        for i, path in enumerate(path_list):
            pv_diagram = extract_pv_slice(cube, path)
            aspect_ratio = float(pv_diagram.shape[1]) / pv_diagram.shape[0]
            ww = WCS(pv_diagram.header)
            ax_pv = plt.subplot(2, 4, 2 + i + int(i > 2), projection=ww)

            # 这里用一个简单的 arcsinh 映射
            norm = mcolors.FuncNorm((np.arcsinh, np.sinh))
            ax_pv.imshow(pv_diagram.data, origin='lower',
                         aspect=aspect_ratio, norm=norm, cmap=plt.cm.RdYlBu_r)

            ax_pv.coords[0].set_format_unit(u.arcmin)
            ax_pv.coords[0].tick_params(direction='in')
            ax_pv.coords[0].display_minor_ticks(True)
            ax_pv.coords[1].set_format_unit(u.km / u.s)
            ax_pv.coords[1].tick_params(direction='in')
            ax_pv.coords[1].display_minor_ticks(True)
            # 有的人喜欢 invert_yaxis() 看起来速度往上走
            ax_pv.invert_yaxis()

            ax_pv.set_xlabel("Offset [arcmin]")
            ax_pv.set_ylabel("Velocity [km/s]")

            # 角度标注
            ax_pv.text(0.1, 0.9, f"{ang_list[i]}°", ha='center', va='center',
                       transform=ax_pv.transAxes, size=14)

            # 关键：把 x 轴限制在 ±(minor/2) arcmin，y 轴限制在 vel_min ~ vel_max
            # ax_pv.set_xlim(-offset_radius_arcmin, offset_radius_arcmin)
            # ax_pv.set_ylim(vel_min, vel_max)

        # ============ 5. 统计像素值（可选） ============
        def sum_pixels_in_ellipse(array_2d, cx, cy, major_pix, minor_pix, pa):
            rows, cols = array_2d.shape
            y_indices, x_indices = np.ogrid[:rows, :cols]
            cos_pa = np.cos(pa)
            sin_pa = np.sin(pa)
            x = x_indices - cx
            y = y_indices - cy
            x_rot = cos_pa * x + sin_pa * y
            y_rot = -sin_pa * x + cos_pa * y
            ellipse_mask = ((x_rot**2 / major_pix**2) + (y_rot**2 / minor_pix**2)) <= 1
            return np.nansum(array_2d[ellipse_mask])

        mom0_cube = np.nansum(velocity_range_slice.unmasked_data[:, :, :].value, axis=0)
        sum_mom0 = sum_pixels_in_ellipse(mom0_cube, center_x, center_y,
                                         ell_major_pix, ell_minor_pix,
                                         ell_pa + 90.0)

        # ============ 6. 保存并结束 ============
        plt.savefig(plot_name)
        plt.close(fig)

        # 这里可返回计算质量/动能等，目前返回 0 占位
        return 0








def bubble_pv_plot_v7(cube, theta, plot_paths=False, plot_name='test.pdf', ellipse={}, 
                      reg_id='', collide=False, reg_file=None, plot_extra_ellipses=False):
    wcs = cube.wcs.celestial
    center_x, center_y = cube.shape[2] / 2, cube.shape[1] / 2
    length = int(np.min([cube.shape[2], cube.shape[1]]) * 0.8)

    central_slice = cube[int(cube.shape[0] / 2) - 1: int(cube.shape[0] / 2) + 1, :, :]
    velocity_range_slice = cube.spectral_slab(
        np.nanmin(ellipse['velocities']) * u.km / u.s,
        np.nanmax(ellipse['velocities']) * u.km / u.s)

    max_val, min_val = max_min_values_in_ellipse(central_slice, ellipse, wcs)

    ell_major = ellipse['major'] / 5.0 * 2.0
    ell_minor = ellipse['minor'] / 5.0 * 2.0
    ell_pa = ellipse['pa'] - 90.
    half_box = 2.3 * ell_major / 2.

    if plot_paths:
        fig = plt.figure(figsize=(16, 8))
        fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97)

        ax1 = plt.subplot(2, 4, 1, projection=wcs)
        cube_moment = central_slice.moment(order=0).value
        ax1.imshow(cube_moment, cmap=plt.cm.gray, interpolation='nearest',
                   origin='lower', aspect='equal',
                   norm=mcolors.PowerNorm(gamma=1.0, vmax=max_val, vmin=0))
        ax1.set_xlabel(f"Right Ascension [{cube.wcs.wcs.radesys}]")
        ax1.set_ylabel(f"Declination [{cube.wcs.wcs.radesys}]")
        ax1.coords[0].tick_params(direction='in')
        ax1.coords[1].tick_params(direction='in')
        ax1.set_xlim(center_x - half_box, center_x + half_box)
        ax1.set_ylim(center_y - half_box, center_y + half_box)

        ellipse_patch = patches.Ellipse(
            xy=(center_x, center_y), width=ell_minor, height=ell_major, 
            angle=ell_pa, edgecolor='y', fc='None', lw=2, ls='--'
        )
        ax1.add_patch(ellipse_patch)

        id_color = 'r' if collide else 'k'
        ax1.text(0.1, 0.9, reg_id, transform=ax1.transAxes, size=18,
                 horizontalalignment='center', verticalalignment='center',
                 color=id_color,
                 path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

        path_list = []
        end_point_list = []
        ang_list = []
        for angle in range(6):
            angle_deg = angle * 30
            phi = theta + angle_deg
            ang_list.append(phi)
            endpoints = calculate_relative_endpoints(center_x, center_y, phi, length)
            end_point_list.append(endpoints)
            path_list.append(create_pv_path(endpoints))

            x_coords, y_coords = zip(*endpoints)
            ax1.plot(x_coords, y_coords, color=(0.5, 0.5, 0.8, 0.9), ls='--')

            # 添加角度文字标注（黑字白边）
            text_pos = np.array(endpoints[0]) * 0.9 + np.array(endpoints[1]) * 0.1
            ax1.text(*text_pos, f"{phi:.0f}$^\\circ$", fontsize=12, ha='center', va='center',
                     color='k',
                     path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

        ax5 = plt.subplot(2, 4, 5, projection=wcs)
        cube_vrange_moment = velocity_range_slice.moment(order=0).value
        max_val_vrange, min_val_vrange = max_min_values_in_ellipse(velocity_range_slice, ellipse, wcs)
        ax5.imshow(cube_vrange_moment, cmap=plt.cm.gray, interpolation='nearest',
                   origin='lower', aspect='equal',
                   vmin=min_val_vrange, vmax=max_val_vrange)
        ax5.set_xlabel(f"Right Ascension [{cube.wcs.wcs.radesys}]")
        ax5.set_ylabel(f"Declination [{cube.wcs.wcs.radesys}]")
        ax5.coords[0].tick_params(direction='in')
        ax5.coords[1].tick_params(direction='in')
        ax5.set_xlim(center_x - half_box, center_x + half_box)
        ax5.set_ylim(center_y - half_box, center_y + half_box)

        ellipse_patch2 = patches.Ellipse(
            xy=(center_x, center_y), width=ell_minor, height=ell_major, 
            angle=ell_pa, edgecolor='y', fc='None', lw=1, ls='--', alpha=0.6
        )
        ax5.add_patch(ellipse_patch2)

        if plot_extra_ellipses and reg_file is not None:
            regions = read_ds9_ellipse_regions(reg_file)
            pix_scales = proj_plane_pixel_scales(cube.wcs.celestial) * 3600
            for reg in regions:
                sc = SkyCoord(ra=reg['ra'], dec=reg['dec'], unit=(u.hourangle, u.deg), frame=FK5(equinox='J2000'))
                px, py = cube.wcs.celestial.wcs_world2pix([[sc.ra.deg, sc.dec.deg]], 1)[0]
                width_pix = reg['major'] / pix_scales[0]
                height_pix = reg['minor'] / pix_scales[1]
                angle_patch = reg['pa'] - 90.0

                epatch1 = patches.Ellipse(xy=(px, py), width=width_pix, height=height_pix,
                                          angle=angle_patch, edgecolor='red', fc='None', lw=2)
                epatch2 = patches.Ellipse(xy=(px, py), width=width_pix, height=height_pix,
                                          angle=angle_patch, edgecolor='red', fc='None', lw=2)
                ax1.add_patch(epatch1)
                ax5.add_patch(epatch2)
                ax1.text(px, py, reg['label'], color='red', fontsize=10, ha='center', va='center')
                ax5.text(px, py, reg['label'], color='red', fontsize=10, ha='center', va='center')

        for i, path in enumerate(path_list):
            pv_diagram = extract_pv_slice(cube, path)
            aspect_ratio = float(pv_diagram.shape[1] / pv_diagram.shape[0])
            ww = WCS(pv_diagram.header)
            ax = plt.subplot(2, 4, 2 + i + int(i > 2), projection=ww)
            norm = FuncNorm((np.arcsinh, np.sinh))
            ax.imshow(pv_diagram.data, aspect=aspect_ratio, origin='lower',
                      norm=norm, cmap=plt.cm.RdYlBu_r)
            ax.coords[0].set_format_unit(u.arcmin)
            ax.coords[0].tick_params(direction='in')
            ax.coords[1].set_format_unit(u.km / u.s)
            ax.coords[1].tick_params(direction='in')
            ax.invert_yaxis()
            ax.set_xlabel("Offset [arcmin]")
            ax.set_ylabel("Velocity [km/s]")
            ax.text(0.1, 0.9, f"{ang_list[i]:.0f}$^\\circ$", transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='k',
                    path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

            cdelt3 = cube.header['CDELT3'] * 1e-3
            deltav = ellipse['velocities'][0] - ellipse['velocities'][1]
            deltav_pix = abs(deltav / cdelt3)
            v_low = np.nanmean(ax.get_ylim()) - deltav_pix / 2.
            v_high = np.nanmean(ax.get_ylim()) + deltav_pix / 2.

            intersect1, intersect2 = ellipse_line_intersection_improved(
                center_x, center_y, ell_major / 2., ell_minor / 2., ell_pa, ang_list[i] - 90.)
            interdist = np.linalg.norm(intersect1 - intersect2)
            p_low = np.nanmean(ax.get_xlim()) - interdist / 2.
            p_high = np.nanmean(ax.get_xlim()) + interdist / 2.

            ax.plot([p_low, p_low], [v_low, v_high], color='gray', ls='--', marker='$-$', lw=1.5,
                    path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])
            ax.plot([p_high, p_high], [v_low, v_high], color='gray', ls='--', marker='$-$', lw=1.5,
                    path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

            if abs(ang_list[i] - ellipse['v_ang']) < 1.:
                p_center = np.nanmean(ax.get_xlim())
                v1_pix = (ellipse['v1'] * 1e3 - cube.header['CRVAL3']) / cube.header['CDELT3'] + cube.header['CRPIX3'] - 1
                v2_pix = (ellipse['v2'] * 1e3 - cube.header['CRVAL3']) / cube.header['CDELT3'] + cube.header['CRPIX3'] - 1
                ax.plot([p_center, p_center], [v1_pix, v2_pix], color='r', ls='--', lw=1.5,
                        path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

        plt.savefig(plot_name)
        plt.close(fig)




























# import matplotlib.pyplot as plt
# from matplotlib import patches, patheffects as pe
# from matplotlib import colors as mcolors
# from astropy import units as u
# from astropy.coordinates import SkyCoord, FK5
# from astropy.wcs import WCS
# from astropy.visualization import FuncNorm
# from astropy.wcs.utils import proj_plane_pixel_scales
# from concurrent.futures import ThreadPoolExecutor
# import numpy as np

def bubble_pv_plot_v8(cube, theta, plot_paths=False, plot_name='test.pdf', ellipse={}, 
                      reg_id='', collide=False, reg_file=None, plot_extra_ellipses=False):

    wcs = cube.wcs.celestial
    center_x, center_y = cube.shape[2] / 2, cube.shape[1] / 2
    length = int(np.min([cube.shape[2], cube.shape[1]]) * 0.8)

    central_slice = cube[int(cube.shape[0] / 2) - 1: int(cube.shape[0] / 2) + 1, :, :]
    velocity_range_slice = cube.spectral_slab(
        np.nanmin(ellipse['velocities']) * u.km / u.s,
        np.nanmax(ellipse['velocities']) * u.km / u.s)

    max_val, min_val = max_min_values_in_ellipse(central_slice, ellipse, wcs)

    ell_major = ellipse['major'] / 5.0 * 2.0
    ell_minor = ellipse['minor'] / 5.0 * 2.0
    ell_pa = ellipse['pa'] - 90.
    half_box = 2.3 * ell_major / 2.

    if plot_paths:
        fig = plt.figure(figsize=(16, 8))
        fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97)

        def plot_mom0(ax, moment_data, maxv, minv):
            ax.imshow(moment_data, cmap=plt.cm.gray, interpolation='nearest',
                      origin='lower', aspect='equal',
                      norm=mcolors.PowerNorm(gamma=1.0, vmax=maxv, vmin=minv))
            ax.set_xlabel(f"Right Ascension [{cube.wcs.wcs.radesys}]")
            ax.set_ylabel(f"Declination [{cube.wcs.wcs.radesys}]")
            ax.coords[0].tick_params(direction='in')
            ax.coords[1].tick_params(direction='in')
            ax.set_xlim(center_x - half_box, center_x + half_box)
            ax.set_ylim(center_y - half_box, center_y + half_box)

        ax1 = plt.subplot(2, 4, 1, projection=wcs)
        ax5 = plt.subplot(2, 4, 5, projection=wcs)

        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(central_slice.moment, order=0)
            f2 = executor.submit(velocity_range_slice.moment, order=0)
            moment1 = f1.result().value
            moment2 = f2.result().value
            max_val_vrange, min_val_vrange = max_min_values_in_ellipse(velocity_range_slice, ellipse, wcs)
            executor.submit(plot_mom0, ax1, moment1, max_val, 0)
            executor.submit(plot_mom0, ax5, moment2, max_val_vrange, min_val_vrange)

        ellipse_patch = patches.Ellipse((center_x, center_y), ell_minor, ell_major,
                                        angle=ell_pa, edgecolor='y', fc='None', lw=2, ls='--')
        ax1.add_patch(ellipse_patch)

        id_color = 'r' if collide else 'k'
        ax1.text(0.1, 0.9, reg_id, transform=ax1.transAxes, size=18,
                 ha='center', va='center', color=id_color,
                 path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

        path_list, end_point_list, ang_list = [], [], []
        for angle in range(6):
            phi = theta + angle * 30
            ang_list.append(phi)
            endpoints = calculate_relative_endpoints(center_x, center_y, phi, length)
            end_point_list.append(endpoints)
            path_list.append(create_pv_path(endpoints))
            x_coords, y_coords = zip(*endpoints)
            ax1.plot(x_coords, y_coords, color=(0.5, 0.5, 0.8, 0.9), ls='--')
            text_pos = np.array(endpoints[0]) * 0.9 + np.array(endpoints[1]) * 0.1
            ax1.text(*text_pos, f"{phi:.0f}$^\\circ$", fontsize=12, ha='center', va='center',
                     color='k', path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

        ellipse_patch2 = patches.Ellipse((center_x, center_y), ell_minor, ell_major,
                                         angle=ell_pa, edgecolor='y', fc='None', lw=1, ls='--', alpha=0.6)
        ax5.add_patch(ellipse_patch2)

        if plot_extra_ellipses and reg_file is not None:
            regions = read_ds9_ellipse_regions(reg_file)
            pix_scales = proj_plane_pixel_scales(cube.wcs.celestial) * 3600
            for reg in regions:
                sc = SkyCoord(ra=reg['ra'], dec=reg['dec'], unit=(u.hourangle, u.deg), frame=FK5(equinox='J2000'))
                px, py = cube.wcs.celestial.wcs_world2pix([[sc.ra.deg, sc.dec.deg]], 1)[0]
                width_pix = reg['major'] / pix_scales[0]
                height_pix = reg['minor'] / pix_scales[1]
                angle_patch = reg['pa'] - 90.0
                epatch1 = patches.Ellipse((px, py), width_pix, height_pix, angle=angle_patch,
                                          edgecolor='red', fc='None', lw=2)
                epatch2 = patches.Ellipse((px, py), width_pix, height_pix, angle=angle_patch,
                                          edgecolor='red', fc='None', lw=2)
                ax1.add_patch(epatch1)
                ax5.add_patch(epatch2)
                ax1.text(px, py, reg['label'], color='red', fontsize=10, ha='center', va='center')
                ax5.text(px, py, reg['label'], color='red', fontsize=10, ha='center', va='center')

        def plot_pv(i):
            path = path_list[i]
            pv_diagram = extract_pv_slice(cube, path)
            aspect_ratio = pv_diagram.shape[1] / pv_diagram.shape[0]
            ww = WCS(pv_diagram.header)
            ax = plt.subplot(2, 4, 2 + i + int(i > 2), projection=ww)
            norm = FuncNorm((np.arcsinh, np.sinh))
            ax.imshow(pv_diagram.data, aspect=aspect_ratio, origin='lower',
                      norm=norm, cmap=plt.cm.RdYlBu_r)
            ax.coords[0].set_format_unit(u.arcmin)
            ax.coords[1].set_format_unit(u.km / u.s)
            ax.coords[0].tick_params(direction='in')
            ax.coords[1].tick_params(direction='in')
            ax.invert_yaxis()
            ax.set_xlabel("Offset [arcmin]")
            ax.set_ylabel("Velocity [km/s]")
            ax.text(0.1, 0.9, f"{ang_list[i]:.0f}$^\\circ$", transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='k',
                    path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

            cdelt3 = cube.header['CDELT3'] * 1e-3
            deltav_pix = abs((ellipse['velocities'][0] - ellipse['velocities'][1]) / cdelt3)
            v_mean = np.nanmean(ax.get_ylim())
            v_low, v_high = v_mean - deltav_pix / 2., v_mean + deltav_pix / 2.

            intersect1, intersect2 = ellipse_line_intersection_improved(
                center_x, center_y, ell_major / 2., ell_minor / 2., ell_pa, ang_list[i] - 90.)
            interdist = np.linalg.norm(intersect1 - intersect2)
            p_mean = np.nanmean(ax.get_xlim())
            p_low, p_high = p_mean - interdist / 2., p_mean + interdist / 2.

            ax.plot([p_low, p_low], [v_low, v_high], color='gray', ls='--', marker='$-$', lw=1.5,
                    path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])
            ax.plot([p_high, p_high], [v_low, v_high], color='gray', ls='--', marker='$-$', lw=1.5,
                    path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

            if abs(ang_list[i] - ellipse['v_ang']) < 1.:
                p_center = np.nanmean(ax.get_xlim())
                v1_pix = (ellipse['v1'] * 1e3 - cube.header['CRVAL3']) / cube.header['CDELT3'] + cube.header['CRPIX3'] - 1
                v2_pix = (ellipse['v2'] * 1e3 - cube.header['CRVAL3']) / cube.header['CDELT3'] + cube.header['CRPIX3'] - 1
                ax.plot([p_center, p_center], [v1_pix, v2_pix], color='r', ls='--', lw=1.5,
                        path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

        with ThreadPoolExecutor(max_workers=6) as executor:
            executor.map(plot_pv, range(6))

        plt.savefig(plot_name)
        plt.close(fig)


# import matplotlib.pyplot as plt
# from matplotlib import patches, patheffects as pe
# from matplotlib import colors as mcolors
# from astropy import units as u
# from astropy.coordinates import SkyCoord, FK5
# from astropy.wcs import WCS
# from astropy.visualization import FuncNorm
# from astropy.wcs.utils import proj_plane_pixel_scales
# from concurrent.futures import ThreadPoolExecutor
# import numpy as np

def bubble_pv_plot_v9(cube, theta, plot_paths=False, plot_name='test.pdf', ellipse={}, 
                      reg_id='', collide=False, reg_file=None, plot_extra_ellipses=False):

    wcs = cube.wcs.celestial
    center_x, center_y = cube.shape[2] / 2, cube.shape[1] / 2
    length = int(np.min([cube.shape[2], cube.shape[1]]) * 0.8)

    central_slice = cube[int(cube.shape[0] / 2) - 1: int(cube.shape[0] / 2) + 1, :, :]
    velocity_range_slice = cube.spectral_slab(
        np.nanmin(ellipse['velocities']) * u.km / u.s,
        np.nanmax(ellipse['velocities']) * u.km / u.s)

    max_val, min_val = max_min_values_in_ellipse(central_slice, ellipse, wcs)

    ell_major = ellipse['major'] / 5.0 * 2.0
    ell_minor = ellipse['minor'] / 5.0 * 2.0
    ell_pa = ellipse['pa'] - 90.
    half_box = 1.9 * ell_major / 2.

    if plot_paths:
        fig = plt.figure(figsize=(16, 8))
        fig.subplots_adjust(left=0.05, right=0.97, bottom=0.05, top=0.97)

        # ==== ax1: Moment-0 map for central slice ====
        ax1 = plt.subplot(2, 4, 1, projection=wcs)
        moment1 = central_slice.moment(order=0).value
        ax1.imshow(moment1, cmap=plt.cm.gray, interpolation='nearest',
                   origin='lower', aspect='equal',
                   norm=mcolors.PowerNorm(gamma=1.0, vmax=max_val, vmin=0))
        ax1.set_xlabel("Right Ascension")
        ax1.set_ylabel("Declination")
        ax1.coords[0].tick_params(direction='in')
        ax1.coords[1].tick_params(direction='in')
        ax1.set_xlim(center_x - half_box, center_x + half_box)
        ax1.set_ylim(center_y - half_box, center_y + half_box)

        ellipse_patch = patches.Ellipse((center_x, center_y), ell_minor, ell_major,
                                        angle=ell_pa, edgecolor='y', fc='None', lw=1, ls='--', alpha=0.6)
        ax1.add_patch(ellipse_patch)

        id_color = 'r' if collide else 'k'
        ax1.text(0.1, 0.9, reg_id, transform=ax1.transAxes, size=18,
                 ha='center', va='center', color=id_color,
                 path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

        path_list, ang_list = [], []
        for angle in range(6):
            phi = theta + angle * 30
            ang_list.append(phi)
            endpoints = calculate_relative_endpoints(center_x, center_y, phi, length)
            path_list.append(create_pv_path(endpoints))
            x, y = zip(*endpoints)
            ax1.plot(x, y, color=(0.5, 0.5, 0.8, 0.9), ls='--')
            text_pos = np.array(endpoints[0]) * 0.9 + np.array(endpoints[1]) * 0.1
            ax1.text(*text_pos, f"{phi:.0f}$^\\circ$", fontsize=12, ha='center', va='center',
                     color='k', path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

        # ==== ax5: Moment-0 map for velocity range ====
        ax5 = plt.subplot(2, 4, 5, projection=wcs)
        moment2 = velocity_range_slice.moment(order=0).value
        max_val_vrange, min_val_vrange = max_min_values_in_ellipse(velocity_range_slice, ellipse, wcs)
        ax5.imshow(moment2, cmap=plt.cm.gray, interpolation='nearest',
                   origin='lower', aspect='equal', vmin=min_val_vrange, vmax=max_val_vrange)
        ax5.set_xlabel("Right Ascension")
        ax5.set_ylabel("Declination")
        ax5.coords[0].tick_params(direction='in')
        ax5.coords[1].tick_params(direction='in')
        ax5.set_xlim(center_x - half_box, center_x + half_box)
        ax5.set_ylim(center_y - half_box, center_y + half_box)

        ellipse_patch2 = patches.Ellipse((center_x, center_y), ell_minor, ell_major,
                                         angle=ell_pa, edgecolor='y', fc='None', lw=1, ls='--', alpha=0.6)
        ax5.add_patch(ellipse_patch2)

        if plot_extra_ellipses and reg_file is not None:
            regions = read_ds9_ellipse_regions(reg_file)
            pix_scales = proj_plane_pixel_scales(cube.wcs.celestial) * 3600
            for reg in regions:
                sc = SkyCoord(ra=reg['ra'], dec=reg['dec'], unit=(u.hourangle, u.deg), frame=FK5(equinox='J2000'))
                px, py = cube.wcs.celestial.wcs_world2pix([[sc.ra.deg, sc.dec.deg]], 1)[0]
                width_pix = reg['major'] / pix_scales[0]
                height_pix = reg['minor'] / pix_scales[1]
                angle_patch = reg['pa'] - 90.0
                epatch1 = patches.Ellipse((px, py), width_pix, height_pix, angle=angle_patch,
                                          edgecolor='red', fc='None', lw=2)
                epatch2 = patches.Ellipse((px, py), width_pix, height_pix, angle=angle_patch,
                                          edgecolor='red', fc='None', lw=2)
                ax1.add_patch(epatch1)
                ax5.add_patch(epatch2)
                ax1.text(px, py, reg['label'], color='red', fontsize=10, ha='center', va='center')
                ax5.text(px, py, reg['label'], color='red', fontsize=10, ha='center', va='center')

        # ==== 提取 PV 数据：用线程池并行提取 ====
        def extract_pv(i):
            return extract_pv_slice(cube, path_list[i])
        with ThreadPoolExecutor(max_workers=6) as executor:
            pv_slices = list(executor.map(extract_pv, range(6)))

        # ==== 绘制 6 个 PV 图 ====
        for i in range(6):
            pv = pv_slices[i]
            ax = plt.subplot(2, 4, 2 + i + int(i > 2), projection=WCS(pv.header))
            norm = FuncNorm((np.arcsinh, np.sinh))
            v_high = (ellipse['velocities'][0] * 1e3 - cube.header['CRVAL3']) / cube.header['CDELT3'] + (cube.header['CRPIX3'] - 1)
            v_low= (ellipse['velocities'][1] * 1e3 - cube.header['CRVAL3']) / cube.header['CDELT3'] + (cube.header['CRPIX3'] - 1)
            # ==== 截取 v_low ~ v_high 的像素行，用于计算 vmin/vmax ====
            v0 = max(0, int(v_low))
            v1 = min(pv.data.shape[0], int(v_high))
            pv_slice_data = pv.data[v0:v1, :]

            # ==== 计算在关心速度范围内的最大最小值 ====
            vmin_clip = np.nanmin(pv_slice_data)
            vmax_clip = np.nanmax(pv_slice_data)

            # ==== 使用 arcsinh 拉伸，但限制在 clip 范围 ====
            norm = FuncNorm((np.arcsinh, np.sinh), vmin=vmin_clip, vmax=vmax_clip)

            # ==== 绘图 ====
            ax.imshow(pv.data, aspect=pv.shape[1]/pv.shape[0], origin='lower',
                      cmap='RdYlBu_r', norm=norm)
            # ax.imshow(pv.data, aspect=pv.shape[1]/pv.shape[0], origin='lower', cmap='RdYlBu_r', norm=norm)
            ax.coords[0].set_format_unit(u.arcmin)
            ax.coords[1].set_format_unit(u.km/u.s)
            ax.coords[0].tick_params(direction='in')
            ax.coords[1].tick_params(direction='in')
            ax.invert_yaxis()
            ax.set_xlabel("Offset [arcmin]")
            ax.set_ylabel("Velocity [km/s]")
            ax.text(0.1, 0.9, f"{ang_list[i]:.0f}$^\\circ$", transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='k',
                    path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

            # Add spatial/velocity window lines
            cdelt3 = cube.header['CDELT3'] * 1e-3
            deltav_pix = abs((ellipse['velocities'][0] - ellipse['velocities'][1]) / cdelt3)
            v_mean = np.nanmean(ax.get_ylim())
            v_low, v_high = v_mean - deltav_pix/2., v_mean + deltav_pix/2.

            inter1, inter2 = ellipse_line_intersection_improved(center_x, center_y, ell_major/2., ell_minor/2., ell_pa, ang_list[i]-90)
            dist = np.linalg.norm(inter1 - inter2)
            p_mean = np.nanmean(ax.get_xlim())
            p_low, p_high = p_mean - dist/2., p_mean + dist/2.

            ax.plot([p_low, p_low], [v_low, v_high], color='gray', ls='--', lw=1.5, marker = '$-$',
                    path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])
            ax.plot([p_high, p_high], [v_low, v_high], color='gray', ls='--', lw=1.5, marker = '$-$',
                    path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])

            if abs(ang_list[i] - ellipse['v_ang']) < 1.:
                p_center = np.nanmean(ax.get_xlim())
                v1_pix = (ellipse['v1'] * 1e3 - cube.header['CRVAL3']) / cube.header['CDELT3'] + (cube.header['CRPIX3'] - 1)
                v2_pix = (ellipse['v2'] * 1e3 - cube.header['CRVAL3']) / cube.header['CDELT3'] + (cube.header['CRPIX3'] - 1)
                ax.plot([p_center, p_center, p_center], [v1_pix, np.nanmean([v1_pix,v2_pix]), v2_pix], color='r', ls='--', lw=1.5, marker = '$-$',
                        path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()])
                # print([v1_pix, np.nanmean([v1_pix,v2_pix]), v2_pix, v_low, v_high])

        plt.savefig(plot_name)
        plt.close(fig)











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
        'id': [],
        'ra_pix': [],
        'dec_pix': [],
        'major': [],
        'minor': [],
        'pa': [],
    }

    for idx, ellipse in enumerate(real_ellipse_dict):
        # Convert (x, y) to (ra, dec)
        if 0:
            ra_dec = wcs.all_pix2world([[ellipse['coorx'], ellipse['coory']]], 1)[0]
            ra, dec = ra_dec

            # Convert to SkyCoord to format as hms and dms
            coord = SkyCoord(ra=ra, dec=dec, unit='deg')
            ra_hms = coord.ra.to_string(unit='hourangle', sep=':', precision=1)
            dec_dms = coord.dec.to_string(unit='deg', sep=':', precision=1, alwayssign=True)

        data['id'].append(idx + 1)
        data['ra_pix'].append(round(ellipse['coorx']))
        data['dec_pix'].append(round(ellipse['coory']))
        data['major'].append(round(ellipse['major']))
        data['minor'].append(round(ellipse['minor']))
        data['pa'].append(round(ellipse['pa']))

    return pd.DataFrame(data)

def save_dataframe_to_fixed_width(df, filename):
    with open(filename, 'w') as file:
        df.to_string(file, index=False, justify='left')

def to_fwf(df, fname):
    content = tabulate(df.values.tolist(), list(df.columns), tablefmt="plain")
    open(fname, "w").write(content)

pd.DataFrame.to_fwf = to_fwf

def reg2df(fits_image_path, ds9_reg_file_path, output_file_path):

    # Read the FITS image to get the WCS
    with fits.open(fits_image_path) as hdul:
        wcs = WCS(hdul[0].header)

    # Read the DS9 region file
    ellipses = read_ds9_reg_file_to_df(ds9_reg_file_path)

    # Generate the DataFrame with synthetic ellipses
    df = generate_synthetic_ellipses_dataframe(ellipses, wcs)

    # Save the DataFrame to a fixed-width file
    save_dataframe_to_fixed_width(df, output_file_path)


def normalize_ellipses(df_path,output_file_path):
    """
    处理椭圆参数：
    1. 如果 major < minor，交换 major 和 minor，并将 pa 加 90。
    2. 将 pa 归一化到 [0, 180) 范围内，考虑对称性（pa ±180° 等价）。
    
    参数：
        df : pd.DataFrame，必须包含 'major', 'minor', 'pa' 三列

    返回：
        pd.DataFrame：处理后的 dataframe（原地修改）
    """
    # 处理 major < minor 的情况
    df = pd.read_fwf(df_path, header=0, infer_nrows=int(1e6))
    swap_mask = df['major'] < df['minor']
    df.loc[swap_mask, ['major', 'minor']] = df.loc[swap_mask, ['minor', 'major']].values
    df.loc[swap_mask, 'pa'] += 90

    # 将 pa 标准化到 [0, 180) 范围内
    df['pa'] = df['pa'] % 180

    save_dataframe_to_fixed_width(df, output_file_path)

    return df

def add_ra_dec_strings(df_path, fits_path, output_file_path):
    """
    使用 FITS 文件中的 WCS 将 df 中的 ra_pix, dec_pix 坐标转换为天球坐标，
    并添加 ra_hms 和 dec_dms 两列（字符串形式，适合打印或保存）。

    参数：
        df : pd.DataFrame，必须包含 'ra_pix' 和 'dec_pix'
        fits_path : str，包含 WCS 的 FITS 文件路径
    
    返回：
        pd.DataFrame：添加了 'ra_hms' 和 'dec_dms' 两列
    """
    # 读取 WCS
    with fits.open(fits_path) as hdul:
        wcs = WCS(hdul[0].header)
    df = pd.read_fwf(df_path, header=0, infer_nrows=int(1e6))
    # 批量转换像素坐标为世界坐标（RA/Dec in deg）
    world_coords = wcs.all_pix2world(df[['ra_pix', 'dec_pix']].values, 1)
    ra_deg, dec_deg = world_coords[:, 0], world_coords[:, 1]

    # 转换为 SkyCoord 对象，格式化为字符串
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    df['ra_hms'] = coords.ra.to_string(unit='hourangle', sep=':', precision=1)
    df['dec_dms'] = coords.dec.to_string(unit='deg', sep=':', precision=1, alwayssign=True)

    save_dataframe_to_fixed_width(df, output_file_path)

    return df

def major_minor_2arcsec(df_path, output_file_path, factor = 1):
    """
    multiply a factor that makes major and minor to arcsec and 
    two colomns maj_as, min_as added. 
    """
    # 读取 WCS
    df = pd.read_fwf(df_path, header=0, infer_nrows=int(1e6))
    df['maj_as'] = df['major'] * factor
    df['min_as'] = df['minor'] * factor
    save_dataframe_to_fixed_width(df, output_file_path)

    return df

def pandas2ds9(ascii_file_path,ds9_file_path):

    def row_to_ds9_region(row):
        return f"ellipse({row['ra_hms']},{row['dec_dms']},{row['maj_as']}\",{row['min_as']}\",{row['pa']}) # text={{{row['id']}}}"

    # Path to the ASCII file containing the table
    # ascii_file_path = "output_fixed_width.txt"

    # Read the file into a DataFrame
    # Make sure to replace '/path/to/your/table_data.txt' with the actual file path
    df = pd.read_fwf(ascii_file_path, header=0, infer_nrows=int(1e6))

    # Apply the function to each row to create DS9 regions
    ds9_regions = df.apply(row_to_ds9_region, axis=1)

    # Save the regions to a new DS9 region file
    # ds9_file_path = "mimictest.reg"
    with open(ds9_file_path, "w") as file:
        file.write("global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n")
        file.write("fk5\n")
        for region in ds9_regions:
            file.write(region + "\n")

def reassign_ids_by_dec(df_path, output_file_path, id_col='id', sort_col='dec_pix'):
    """
    按照指定列（默认 dec_pix）升序排序 dataframe，
    然后重新分配一个连续的 id（从 1 开始）

    参数：
        df : pd.DataFrame
        id_col : str，新 id 的列名（默认 'id'）
        sort_col : str，用于排序的列名（默认 'dec_pix'）

    返回：
        新的 dataframe，包含重新编号的 id 列（排在最前面）
    """
    df = pd.read_fwf(df_path, header=0, infer_nrows=int(1e6))    

    # 按 sort_col 排序
    df_sorted = df.sort_values(by=sort_col).reset_index(drop=True)

    # 添加新的连续 id，从 1 开始
    df_sorted[id_col] = np.arange(1, len(df_sorted) + 1)

    # 将 id 列移到最前面
    cols = [id_col] + [col for col in df_sorted.columns if col != id_col]
    df_sorted = df_sorted[cols]
    save_dataframe_to_fixed_width(df_sorted, output_file_path)
    return df_sorted



def selectAccordingToColumnValue(df_path, output_file_path, target_col='good', target_value='pv'):
    """
    Select rows from a DataFrame where target_col == target_value,
    preserving original order, and save to output_file_path.

    Parameters:
    -----------
    df_path : str
        Path to the input tab file.
    output_file_path : str
        Path to the output fixed-width format file.
    target_col : str
        Column name to filter on. Default is 'good'.
    target_value : str
        Value to match in the column. Default is 'pv'.
    """
    df = pd.read_fwf(df_path, header=0, infer_nrows=int(1e6))
    df_selected = df[df[target_col] == target_value]
    save_dataframe_to_fixed_width(df_selected, output_file_path)





import pandas as pd

def compute_center_velocity(df_path, output_file_path):
    """
    Compute center velocity as abs((v1 + v2) / 2) and save result.

    Parameters:
    -----------
    df_path : str
        Path to the input tab file.
    output_file_path : str
        Path to the output fixed-width format file.
    """
    df = pd.read_fwf(df_path, header=0, infer_nrows=int(1e6))
    df['center_vel'] = ((df['v1'] + df['v2']) / 2)
    save_dataframe_to_fixed_width(df, output_file_path)


def compute_expansion_velocity(df_path, output_file_path):
    """
    Compute expansion velocity as abs((v2 - v1) / 2) and save result.

    Parameters:
    -----------
    df_path : str
        Path to the input tab file.
    output_file_path : str
        Path to the output fixed-width format file.
    """
    df = pd.read_fwf(df_path, header=0, infer_nrows=int(1e6))
    df['expansion_vel'] = ((df['v2'] - df['v1']) / 2).abs()
    save_dataframe_to_fixed_width(df, output_file_path)


def compute_radius_pc(df_path, output_file_path):
    """
    Compute physical radius in parsec from maj_as using: maj_as / 60 * 216

    Parameters:
    -----------
    df_path : str
        Path to the input tab file.
    output_file_path : str
        Path to the output fixed-width format file.
    """
    df = pd.read_fwf(df_path, header=0, infer_nrows=int(1e6))
    df['radius_pc'] = df['maj_as'] / 60. * 216
    save_dataframe_to_fixed_width(df, output_file_path)




def compute_radius_pc_deconvolve(df_path, output_file_path,beamsize=216.):
    """
    Compute physical radius in parsec from maj_as using: maj_as / 60 * 216

    Parameters:
    -----------
    df_path : str
        Path to the input tab file.
    output_file_path : str
        Path to the output fixed-width format file.
    """
    df = pd.read_fwf(df_path, header=0, infer_nrows=int(1e6))
    df['radius_pc'] = np.sqrt((df['maj_as'] / 60. * 216)**2-beamsize**2)
    save_dataframe_to_fixed_width(df, output_file_path)






















def parse_mixed_format_file(input_path):
    rows = []
    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 15:
                continue

            name = parts[0].strip()
            y_arcmin = parts[1].strip()
            x_arcmin = parts[2].strip()
            ra_dec_part = parts[3].strip()
            tokens = ra_dec_part.split()
            ra_h, ra_m, ra_s = tokens[0:3]
            dec_sign = tokens[3][0]
            dec_deg = tokens[3][1:]
            dec_min, dec_sec = tokens[4:6]
            v_hel = parts[4].strip()
            dv = parts[5].strip()
            fwhmi = parts[6].strip()
            fwhma = parts[7].strip()
            pa = parts[8].strip()
            intk = parts[9].strip()
            contr = parts[10].strip()
            q = parts[11].strip()
            t = parts[12].strip()
            c = parts[13].strip()
            remarks = parts[14].strip()

            rows.append([
                name, x_arcmin, y_arcmin,
                ra_h, ra_m, ra_s,
                dec_sign + dec_deg, dec_min, dec_sec,
                v_hel, dv, fwhmi, fwhma, pa,
                intk, contr, q, t, c, remarks
            ])

    columns = [
        'Name', 'X_arcmin', 'Y_arcmin',
        'RA_h', 'RA_m', 'RA_s',
        'Dec_deg', 'Dec_min', 'Dec_sec',
        'V_Hel', 'DV', 'FWHMI', 'FWHMA', 'PA',
        'IntK', 'Contr', 'Q', 'T', 'C', 'Remarks'
    ]
    return pd.DataFrame(rows, columns=columns)

def process_hi_hole_with_ellipses(input_path, tab_output_path, reg_output_path, pc_per_arcsec=3.8):
    df = parse_mixed_format_file(input_path)

    # 拼接 RA/Dec 字符串
    ra_str = df['RA_h'].astype(str) + 'h' + df['RA_m'].astype(str) + 'm' + df['RA_s'].astype(str) + 's'
    dec_str = df['Dec_deg'].astype(str) + 'd' + df['Dec_min'].astype(str) + 'm' + df['Dec_sec'].astype(str) + 's'

    coords_b1950 = SkyCoord(ra=ra_str, dec=dec_str, frame='fk4', equinox='B1950')
    fk5_2000 = FK5(equinox='J2000')
    coords_j2000 = coords_b1950.transform_to(fk5_2000)

    df['ra_hms'] = coords_j2000.ra.to_string(unit=u.hourangle, sep=':', precision=1, pad=True)
    df['dec_dms'] = coords_j2000.dec.to_string(unit=u.deg, sep=':', precision=1, alwayssign=True, pad=True)

    df.drop(columns=['RA_h', 'RA_m', 'RA_s', 'Dec_deg', 'Dec_min', 'Dec_sec'], inplace=True)
    front_cols = ['Name', 'X_arcmin', 'Y_arcmin', 'ra_hms', 'dec_dms']
    df = df[front_cols + [col for col in df.columns if col not in front_cols]]

    save_dataframe_to_fixed_width(df, tab_output_path)
    print(f"[✓] 已保存表格至: {tab_output_path}")

    # 输出 region 文件（椭圆）
    with open(reg_output_path, 'w') as reg:
        reg.write("# Region file format: DS9 version 4.1\n")
        reg.write("global color=red font='helvetica 10 bold' select=1 edit=1 move=1 delete=1 include=1 fixed=0\n")
        reg.write("fk5\n")

        for idx, row in df.iterrows():
            ra = row['ra_hms']
            dec = row['dec_dms']
            fwhma = float(row['FWHMA'])  # major (pc)
            fwhmi = float(row['FWHMI'])  # minor (pc)
            pa = np.array(float(row['PA']))+90        # deg
            v = row['V_Hel']

            # 转换为 arcsec
            major_arcsec = round(fwhma / pc_per_arcsec, 2)
            minor_arcsec = round(fwhmi / pc_per_arcsec, 2)

            # ellipse(ra, dec, major", minor", angle)
            reg.write(f"ellipse({ra},{dec},{major_arcsec}\",{minor_arcsec}\",{pa}) # text={{{idx+1}:{v}}}\n")

    print(f"[✓] 已保存椭圆 region 文件至: {reg_output_path}")