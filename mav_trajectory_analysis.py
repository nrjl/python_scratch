from __future__ import print_function
import argparse
import rosbag
import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button


class SegmentPoly:
    def __init__(self, segment):
        self.px = np.polynomial.Polynomial(segment.x)
        self.py = np.polynomial.Polynomial(segment.y)
        self.pz = np.polynomial.Polynomial(segment.z)
        self.pyaw = np.polynomial.Polynomial(segment.yaw)
        self.t = segment.segment_time.secs + 1.0e-9 * segment.segment_time.nsecs

    def get_trajectory(self, dt=0.1, include_end=True, return_uvw=False):
        tp = np.arange(0.0, self.t, dt)
        if include_end and self.t > tp[-1]:
            tp = np.append(tp, self.t)

        if not return_uvw:
            return self.px(tp), self.py(tp), self.pz(tp), self.pyaw(tp)
        else:
            yaw = self.pyaw(tp)
            u = np.sin(yaw)
            v = np.cos(yaw)
            w = np.zeros_like(u)
            return self.px(tp), self.py(tp), self.pz(tp), u, v, w

    def is_linear(self, threshold=1e-9):
        return all(abs(self.px.coef[2:]) < threshold) and \
               all(abs(self.py.coef[2:]) < threshold) and \
               all(abs(self.pz.coef[2:]) < threshold)

    def is_circular(self, n_points=10, tolerance=1e-3):
        d2x = self.px.deriv(2)
        d2y = self.py.deriv(2)
        # Crappy sampling-based test
        sq_curvature = d2y(0)**2 + d2x(0)**2
        if np.isnan(sq_curvature):
            return False

        for t in np.linspace(0, self.t, n_points):
            if abs(d2y(t)**2 + d2x(t)**2 - sq_curvature) > tolerance:
                return False
        return True

class RadarTrajectoryBag:
    """
    Object for extracting a trajectory from a ROS bag file and identifying transition points
    """

    def __init__(self, bagfile, vehicle_name='eagle'):

        self._bagfile = bagfile
        self._vehicle_name = vehicle_name
        self._current_segments = [0, 1]

        bag = rosbag.Bag(self._bagfile)
        trajectory_topic = self.get_vehicle_topic('command/polynomial_trajectory')
        pose_topic = self.get_vehicle_topic('command/pose')

        # Get the trajectory and commanded poses
        self._n_trajectories = bag.get_message_count(trajectory_topic)
        self._n_poses = bag.get_message_count(pose_topic)
        assert self._n_trajectories == 1, ("Found {0} trajectories on topic {1}, must be exactly 1"
                                           .format(self._n_trajectories, trajectory_topic))

        # Get the trajectory and command pose messages
        self._trajectory = []
        self._poses = []
        for topic, msg, t in bag.read_messages(topics=[trajectory_topic, pose_topic]):
            if topic == trajectory_topic:
                self._trajectory = msg
            elif topic == pose_topic:
                self._poses.append(msg)

        bag.close()

        self.segments = [SegmentPoly(segment) for segment in self._trajectory.segments]
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

    def _keep_segments(self, event):
        print('Saving segments {0} to {1}'.format(self._current_segments[0], self._current_segments[1]))

    def _discard_segments(self, event):
        print('Discarding segments {0} to {1}'.format(self._current_segments[0], self._current_segments[1]))

    def extract_radar_segments(self, dt=0.1, dt_arrow=1.0, arrow_scale=0.5):
        # Accumulate segments
        accumulated_segments = []
        start_seg = 0
        current_seg = 1
        while start_seg < len(self.segments):
            # Check if we need to accumulate

            # Check if linear or circular and accumulate
            if self.segments[start_seg].is_linear() and self.segments[current_seg].is_linear():
                current_seg += 1
                continue

            if self.segments[start_seg].is_circular() and self.segments[current_seg].is_circular():
                current_seg += 1
                continue

            accumulated_segments.append([start_seg, current_seg])
            start_seg = current_seg
            current_seg = start_seg + 1

        # Collect straight line and fixed-curvature together
        fig = plt.figure()
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        axkeep = fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axdiscard = fig.add_axes([0.81, 0.05, 0.1, 0.075])
        bkeep = Button(axkeep, 'Keep')
        bkeep.on_clicked(self._keep_segments)
        bdiscard = Button(axdiscard, 'Discard')
        bdiscard.on_clicked(self._discard_segments)

        # Plot all segments in first axis:
        ax0, seglines, arrows = self.plot_trajectory(dt=0.1, dt_arrow=1.0, arrow_scale=0.5, ax=ax0)

        # Now we have accumulated segments, mark and plot in sequence
        for a_seg in accumulated_segments:
            self._current_segments = a_seg
            ax1.cla()

            for line in seglines:
                line.set_color('b')
            for i in range(a_seg[0], a_seg[1]):
                seglines[i].set_color('r')
                x, y, z, yaw = self.segments[i].get_trajectory()
                ax1.plot(x, y, z, 'r')
            plt.waitforbuttonpress()

    def plot_trajectory(self, dt=0.1, dt_arrow=1.0, arrow_scale=0.5, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('E (m)')
        ax.set_ylabel('N (m)')
        ax.set_zlabel('U (m)')

        seglines = []
        arrows = []

        for segment in self.segments:
            x, y, z, yaw = segment.get_trajectory(dt = dt, include_end=True)
            seglines.append(ax.plot(x, y, z)[0])

            # Plot arrows (even spacing)
            x, y, z, u, v, w = segment.get_trajectory(dt = dt_arrow, include_end=False, return_uvw=True)
            arrows.append(ax.quiver(x, y, z, u*arrow_scale, v*arrow_scale, w))

        return ax, seglines, arrows

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
    bag_obj.extract_radar_segments()
    plt.show()
