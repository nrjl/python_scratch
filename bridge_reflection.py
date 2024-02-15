import numpy as np
import matplotlib.pyplot as plt


def parabola_bridge(x: float, k: float=1.0):
    '''Assume our bridge starts at (0,0,0) and ends at (1,0,0)
    x is the lateral bridge distance x \in [0,1]
    k is the highest point on the bridge
    '''
    return -4*k*x*(x-1)

observer_position = np.array([0.8, -0.5, 0.01])

fh = plt.figure(figsize=plt.figaspect(0.5))
ax0 = fh.add_subplot(1, 2, 1, projection='3d')
xx = np.linspace(0, 1, 100)
yy = np.zeros_like(xx)
zz = parabola_bridge(xx, k=0.02)
ax0.plot(xx, yy, zz, 'b-')
ax0.plot(observer_position[0], observer_position[1], observer_position[2], 'r.')
ax0.set_aspect('equal', adjustable='box')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$y$')
ax0.set_zlabel('$z$')


# Observer distance
bridge_points = np.array([xx, yy, zz]).T
dd = np.sqrt(((bridge_points - observer_position)**2).sum(axis=1))
ax1 = fh.add_subplot(1, 2, 2)
ax1.plot(xx, dd)
ax1.set_xlabel('$x$')
ax1.set_ylabel('$d$')
ax1.set_title('Distance $d$ from observer at {0} to bridge points'.format(observer_position))

plt.show()

