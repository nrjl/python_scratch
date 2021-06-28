import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from mayavi import mlab
import os
import pandas
import argparse
from time import sleep

parser = argparse.ArgumentParser(
    description='Create mayavi volume render for radar data')
parser.add_argument('-d', '--data-dir', default='/media/intel_data2/findmine/10_sar/03_no_reflectors', help='Data directory')
parser.add_argument('-n', '--n-angles', type=int, default=100, help='Number of angles to render')
parser.add_argument('-p', '--percentile', type=int, default=90, help='Percetile return cutoff')
parser.add_argument('--zoom-target', type=int, default=1, help='Target index to zoom in on')
parser.add_argument('--offscreen', action='store_true', help='Offscreen render')
parser.add_argument('-s', '--save-animation', action='store_true', help='Save animation frames')
parser.add_argument('--magnification', default=1, help='Magnification (higher resolution render)')

args = parser.parse_args()
data_dir = args.data_dir
n_angles = args.n_angles
mlab.options.offscreen = args.offscreen

full_data = spio.loadmat(os.path.join(data_dir, 'py_data', 'tx2rx2.mat'))
target_df = pandas.read_csv(os.path.join(data_dir, 'targets.txt'))
output_dir = '/media/intel_data2/findmine/10_sar/03_no_reflectors/py_data/'

x = full_data['x'].squeeze()
y = full_data['y'].squeeze()
z = full_data['z'].squeeze()
depths = full_data['depths'].squeeze()

z0 = z.mean()

# Radar
radar_returns = full_data['radar_returns']
radar_magnitudes = np.absolute(radar_returns)

radar_dB = 20*np.log10(radar_magnitudes)
vmin = 20*np.log10(np.percentile(radar_magnitudes, args.percentile))     # Alex's 90th percentile

fig = mlab.figure(size=(960, 540), bgcolor=(1.0,1.0,1.0))
radar_field = mlab.pipeline.scalar_field(radar_dB)
radar_field.origin = [0.0, 0.0, 0.0]

# radar_field.spacing = [(x[-1]-x[0])/(radar_dB.shape[0]-1), (y[-1]-y[0])/(radar_dB.shape[1]-1), (depths[-1]-depths[0])/(radar_dB.shape[2]-1)]
mlab.pipeline.volume(radar_field, vmin=vmin)
mlab.outline(color=(0.2, 0.2, 0.2))

# mlab.pipeline.image_plane_widget(radar_field, plane_orientation='z_axes', slice_index=1)

# Ugly ugly convert because mayavi doesn't like small spacing for the volume
def convert_position(xi, yi, zi):
    xo = (xi - x[0]) / (x[-1]-x[0])*(radar_dB.shape[0]-1)
    yo = (yi - y[0]) / (y[-1] - y[0]) * (radar_dB.shape[1] - 1)
    zo = (zi -z0 - depths[0]) / (depths[-1] - depths[0]) * (radar_dB.shape[2] - 1)
    return xo, yo, zo

# Targets
tx, ty, tz = convert_position(target_df.x, target_df.y, target_df.z)
h_targets = mlab.points3d(ty, tx, tz, scale_factor=30, mode='cube', opacity=1.0, color=(1.0, 0, 0))
sx, sy, sz = convert_position(x, y, z)
h_surf = mlab.surf(sy, sx, sz.transpose(), colormap='terrain', opacity=1.0, line_width=0.5)

az0, el0, d0, fp0 = 0, 64, 1080, [350, 350, 0]
mlab.view(az0, el0, d0, fp0)

# @mlab.animate(delay=50)
# def anim():
    # Rotate scene and export
i = 0
[az, el, d, fp] = mlab.view()
az_array = np.linspace(0, 360, n_angles)
for az in az_array:
    mlab.view(az, el, d, fp)
    h_surf.actor.property.opacity = max(0.0, (1.0-2.0*float(i)/n_angles))
    h_targets.actor.property.opacity = max(0.0, (1.0-1.5*float(i)/n_angles))
    if args.save_animation:
        mlab.savefig(os.path.join(output_dir, 'output{0:03d}.png'.format(i)),
                     magnification=args.magnification)
        i += 1
    else:
        sleep(0.05)


[az, el, d, fp] = mlab.view()
el_array = np.linspace(el, 5, int(n_angles/2))
d_array = np.linspace(d, 1580, int(n_angles/2))

for el, d in zip(el_array, d_array):
    mlab.view(az, el, d, fp)
    if args.save_animation:
        mlab.savefig(os.path.join(output_dir, 'output{0:03d}.png'.format(i)),
                     magnification=args.magnification)
        i += 1
    else:
        sleep(0.05)

[az, el, d, fp] = mlab.view()
el_array = np.linspace(el, 55, int(n_angles / 2))
d_array = np.linspace(d, 250, int(n_angles / 2))
fp_array = np.linspace(fp, np.array([ty[args.zoom_target], tx[args.zoom_target], tz[args.zoom_target]]), int(n_angles/2))

for el, d, fp in zip(el_array, d_array, fp_array):
    mlab.view(az, el, d, fp)
    if args.save_animation:
        mlab.savefig(os.path.join(output_dir, 'output{0:03d}.png'.format(i)),
                     magnification=args.magnification)
        i += 1
    else:
        sleep(0.05)

az_array = np.linspace(0, 360, int(n_angles/2))
for az in az_array:
    mlab.view(az, el, d, fp)
    if args.save_animation:
        mlab.savefig(os.path.join(output_dir, 'output{0:03d}.png'.format(i)),
                     magnification=args.magnification)
        i += 1
    else:
        sleep(0.05)

[az, el, d, fp] = mlab.view()
az_array = np.linspace(az, az0, int(n_angles / 2))
el_array = np.linspace(el, el0, int(n_angles / 2))
d_array = np.linspace(d, d0, int(n_angles / 2))
fp_array = np.linspace(fp, fp0, int(n_angles/2))
for az, el, d, fp in zip(az_array, el_array, d_array, fp_array):
    mlab.view(az, el, d, fp)
    if args.save_animation:
        mlab.savefig(os.path.join(output_dir, 'output{0:03d}.png'.format(i)),
                     magnification=args.magnification)
        i += 1
    else:
        sleep(0.05)

# anim()
mlab.show()

