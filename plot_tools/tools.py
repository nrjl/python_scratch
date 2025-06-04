import numpy as np

def axes_equal(ax, data=None):
    """
    Method to set equal aspect ratio in 2D or 3D plots

    If no data specified, use axis limits, otherwise use data limits

    Data: nx3 array
    """

    if ax.name == "3d":
        if data is None:    
            data = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]).T
        ax.set_box_aspect(np.ptp(data, axis = 0))
    else:
        ax.set_aspect('equal', 'box')