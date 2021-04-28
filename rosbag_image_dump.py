from __future__ import print_function
import argparse
import rosbag
import rospy
import cv_bridge
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import datetime

def create_dir(path):
    path = os.path.expanduser(path)
    path = os.path.normpath(path)
    if not os.path.exists(path):
        os.makedirs(path)
        # os.makedirs(path, exist_ok=True)
    return path


class BagReader(object):
    """
    Simple bag file reader
    """

    def __init__(self, bagfile):

        print('Reading bag file {0}...'.format(bagfile))
        self.bag = rosbag.Bag(bagfile)
        self.bridge = cv_bridge.CvBridge()

    def __del__(self):
        self.bag.close()

    def _check_topic(self, topic_name, msg_type):
        """Check the topic is in the bag and has the correct message type."""
        types, topic_info = self.bag.get_type_and_topic_info(topic_name)
        assert topic_info[topic_name].msg_type == msg_type
        message_count = topic_info[topic_name].message_count
        assert message_count > 0
        return message_count

    def _build_output_dir(self, out_dir=None, output_extension=None):
        if out_dir is not None:
            dir_out = create_dir(out_dir)
        elif output_extension is not None:
            dir_out = create_dir(os.path.join(self.path_out, output_extension))
        else:
            dir_out = create_dir(self.path_out)
        return dir_out

    def dump_images(self, rgb_topic, out_dir, msg_type='sensor_msgs/Image', quality=95):
        n_images = self._check_topic(rgb_topic, msg_type)

        out_dir_rgb = self._build_output_dir(out_dir, 'rgb')

        with tqdm(total=n_images) as waitbar:
            waitbar.set_description('Extracting images from bag file')

            for i, (topic, msg, t) in enumerate(self.bag.read_messages(rgb_topic)):

                imgtime = datetime.datetime.fromtimestamp(t.secs)
                datestring = imgtime.strftime('%Y%m%d_%H%M%S') + \
                    '_%03d' % (t.nsecs / 1e6)

                cimage = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
                im = Image.fromarray(cimage)
                filename = os.path.join(out_dir, datestring + '.jpg')
                im.save(filename, 'JPEG', quality=quality)
                waitbar.update()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract images from rosbag')
    parser.add_argument('-o', '--output-dir', default='.', help='Output directory')
    parser.add_argument('BAGFILE', type=str, help='Input rosbag file')
    parser.add_argument('IMAGE_TOPIC', type=str, help='ROS image topic')

    args = parser.parse_args()

    bag_obj = BagReader(args.BAGFILE)
    bag_obj.dump_images(args.IMAGE_TOPIC, out_dir=args.output_dir)
