from __future__ import print_function
import argparse
import rosbag
import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RadarTrajectoryBag:
    """
    Object for extracting a trajectory from a ROS bag file and identifying transition points
    """

    def __init__(self, bagfile, vehicle_name='eagle'):

        self._bagfile = bagfile
        self._vehicle_name = vehicle_name

        bag = rosbag.Bag(self._bagfile)
        trajectory_topic = self.get_vehicle_topic('command/polynomial_trajectory')
        pose_topic = self.get_vehicle_topic('command/pose')

        # Get the trajectory and commanded poses
        self._n_trajectories = bag.get_message_count(trajectory_topic)
        self._n_poses = bag.get_message_count(pose_topic)
        assert self._n_trajectories == 1, ("Found {0} trajectories on topic {1}, must be exactly 1"
                                           .format(self._n_trajectories, trajectory_topic))

        # Get the trajectory and command poses
        self._trajectory = []
        self._poses = []
        for topic, msg, t in bag.read_messages(topics=[trajectory_topic, pose_topic]):
            if topic == trajectory_topic:
                self._trajectory = msg
            elif topic == pose_topic:
                self._poses.append(msg)

        bag.close()
        print('Found trajectory with {0} segments, and {1} commanded pose messages'
              .format(len(self._trajectory.segments), len(self._poses)))

    def get_segment_times(self):

        # Segment times in nanoseconds
        segment_durations_nsecs = np.zeros(len(self._trajectory.segments)+1, dtype=int)

        # Extract transition times
        for i, segment in enumerate(self._trajectory.segments):
            segment_durations_nsecs[i+1] = int(segment.segment_time.secs*1e9) + int(segment.segment_time.nsecs)

        segment_times = np.cumsum(segment_durations_nsecs)

        return segment_times

    def plot_trajectory(self, dt=0.1, dt_arrow=1.0):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        seglines = []

        for segment in self._trajectory.segments[0:2]:
            tp = np.arange(0.0, segment.segment_time.secs + 1.0e-9*segment.segment_time.nsecs, dt)
            tp = np.append(tp, segment.segment_time.secs + 1.0e-9*segment.segment_time.nsecs)
            x_poly = np.polynomial.Polynomial(segment.x)
            y_poly = np.polynomial.Polynomial(segment.y)
            z_poly = np.polynomial.Polynomial(segment.z)
            yaw_poly = np.polynomial.Polynomial(segment.yaw)

            seglines.append(ax.plot(x_poly(tp), y_poly(tp), z_poly(tp))[0])

        return fig, ax, seglines

    def get_vehicle_topic(self, topic):
        return '/'+self._vehicle_name+'/'+topic


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract flight trajectory on/off timestamps from a rosbag and export')
    parser.add_argument('-n', '--name', default='eagle', help='Vehicle name')
    parser.add_argument('-o', '--output-dir', default='.', help='Output directory')
    parser.add_argument('bagfile', type=str, help='Input rosbag file')

    args = parser.parse_args()

    bag_obj = RadarTrajectoryBag(args.bagfile, args.name)
    bag_obj.plot_trajectory()
    plt.show()
