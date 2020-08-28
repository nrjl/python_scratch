import os
from shutil import copy2
import argparse
import tqdm

# Based on https://alastaira.wordpress.com/2011/07/06/converting-tms-tile-coordinates-to-googlebingosm-tile-coordinates/

def get_int_dirs(target_dir):

    # Get all subdirectories with integer names
    for root, subdirs, files in os.walk(target_dir):
        int_dirs = []
        for subdir in subdirs:
            try:
                temp = int(subdir)
            except ValueError:
                continue
            int_dirs.append(subdir)
        return int_dirs


def main():
    parser = argparse.ArgumentParser(description='Change TMS aligned tiles to Slippy OSM format')
    parser.add_argument('-i', '--input-dir', required=True, help='Directory of TMS tiles INPUT_DIR (/{z}/{y}/{x}.png)')
    parser.add_argument('-o', '--output-dir', required=True, help='Target directory for OSM output tiles (must be different from input)')
    args = parser.parse_args()

    assert args.input_dir != args.output_dir, 'Input and output directories must be different'

    # Loop over zoom level directories
    for zdir in get_int_dirs(args.input_dir):
        # print('Zoom level {0}'.format(zdir))
        full_old_zdir = os.path.join(args.input_dir, zdir)
        full_new_zdir = os.path.join(args.output_dir, zdir)
        os.makedirs(full_new_zdir, exist_ok=True)
        y_max = 1 << int(zdir)

        for xdir in tqdm.tqdm(get_int_dirs(full_old_zdir), desc='Zoom level {0}'.format(zdir)):
            full_old_xdir = os.path.join(full_old_zdir, xdir)
            full_new_xdir = os.path.join(full_new_zdir, xdir)
            os.makedirs(full_new_xdir, exist_ok=True)

            for file in os.listdir(full_old_xdir):
                if file.endswith('.png'):
                    tsm_y = file.split('.')[0]
                    try:
                        osm_y = y_max - int(tsm_y) - 1
                    except ValueError:
                        print('File {0} not properly parsed, skipping.'.format(os.path.join(full_new_xdir, file)))
                        continue
                    copy2(os.path.join(full_old_xdir, file), os.path.join(full_new_xdir, str(osm_y)+'.png'))


if __name__ == "__main__":
    main()

