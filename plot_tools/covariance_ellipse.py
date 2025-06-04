from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import chi2
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D


def _check_mean_cov(mean, cov, d=2):
    mean = np.array(mean)
    cov = np.array(cov)
    assert (
        len(mean) == cov.shape[0] == cov.shape[1] == d
    ), "len(mean) must = cov.shape[0] = cov.shape[1]"
    return mean, cov


class CovarianceIntervals(ABC):
    """
    Abstract base class for plotting animated confidence intervals (one derived subclass for number of dimensions,
    e.g. 2D ellipses, 3D ellipsoids
    """

    def get_artists(self):
        return self._artists

    @abstractmethod
    def update(self, mean, cov):
        """Update artists for animation"""
        pass

    @abstractmethod
    def get_legend_handle(self):
        """Return an object that can be used by a legend call"""
        pass


class CovarianceEllipses2D(CovarianceIntervals):
    """Handler for plotting and animating 2D covariance ellipses"""

    def __init__(
        self,
        ax,
        mean=[0, 0],
        cov=[[1, 0], [0, 1]],
        ellipse_masses=[0.68, 0.95],
        alpha=0.3,
        animated=False,
        facecolor=None,
        **kwargs
    ) -> None:
        self.mean, self.cov = _check_mean_cov(mean, cov)
        self.ax = ax
        self._ellipse_scales = np.sqrt(chi2.ppf(ellipse_masses, df=2))
        self._artists = []
        d1, d2, angle = self._get_ellipse_params()

        for s in self._ellipse_scales:
            ellipse = Ellipse(
                mean,
                width=d1 * s,
                height=d2 * s,
                angle=angle,
                alpha=alpha,
                animated=animated,
                **kwargs
            )
            if facecolor == "none":
                ellipse.set_facecolor("none")
            ax.add_patch(ellipse)
            self._artists.append(ellipse)

    def update(self, mean, cov):
        self.mean, self.cov = _check_mean_cov(mean, cov)
        d1, d2, angle = self._get_ellipse_params()
        for ell, s in zip(self._artists, self._ellipse_scales):
            ell.set_width(d1 * s)
            ell.set_height(d2 * s)
            ell.set_angle(angle)
            ell.set_center(mean)

        return self._artists

    def _get_ellipse_params(self):
        # Get ellipse parameters (note angle returned in degrees because that's what Ellipse.set_angle() expects)
        # Get eigenvalues and sort them (big first)
        w, V = np.linalg.eig(self.cov)
        idx = -w.argsort()[::-1]
        w = w[idx]
        V = V[:, idx]
        d1, d2 = 2 * np.sqrt(w)
        angle = np.arctan2(V[1, 0], V[0, 0]) * 180.0 / np.pi
        return d1, d2, angle

    def get_legend_handle(self):
        return self._artists[0]


class CovarianceEllipsoids3D(CovarianceIntervals):
    """Plot covariance ellipsoids in 3D
    Based on https://github.com/CircusMonkey/covariance-ellipsoid/blob/master/ellipsoid.py
    """

    def __init__(
        self,
        ax,
        mean=[0, 0, 0],
        cov=np.eye(3),
        ellipse_masses=[0.68, 0.95],
        animated=True,
        num_u: int = 40,
        num_v: int = 20,
        **kwargs
    ) -> None:
        self.mean, self.cov = _check_mean_cov(mean, cov, d=3)
        assert isinstance(
            ax, Axes3D
        ), "Axes must be mpl_toolkits.mplot3d.Axes3D instance"
        self.ax = ax
        self.kwargs = {"color": "b", "edgecolor": "none", "alpha": 0.2}
        self.kwargs.update(kwargs)

        # Calculate the target volume of the ellipsoid with the specified confidence bounds
        self._ellipsoid_radii = np.sqrt(chi2.ppf(ellipse_masses, 3))

        u = np.linspace(0, 2 * np.pi, num_u)
        v = np.linspace(0, np.pi, num_v)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        self._unit_points = np.stack((x.ravel(), y.ravel(), z.ravel()), 0)
        self._points_shape = x.shape

        self._artists = []

        for r in self._ellipsoid_radii:
            ell = self.ax.plot_surface(*self._make_ellipsoid(r), **self.kwargs)
            self._artists.append(ell)

    def _make_ellipsoid(self, radius):
        """Construct ellipsoid for specified mean, cov and equivalent target radius"""

        # Extract the Eigenvalues of the covariance to calculate scale
        scale = radius / (np.linalg.det(self.cov) ** (1.0 / 6))

        X, Y, Z = scale * (self.cov @ self._unit_points).reshape(3, *self._points_shape)
        return X + self.mean[0], Y + self.mean[1], Z + self.mean[2]

    def update(self, mean, cov):
        self.mean, self.cov = _check_mean_cov(mean, cov, d=3)

        for i, r in enumerate(self._ellipsoid_radii):
            # I don't like that you have to completely replot, but seems like the only/best way
            self._artists[i].remove()
            self._artists[i] = self.ax.plot_surface(
                *self._make_ellipsoid(r), **self.kwargs
            )

        return self._artists

    def get_legend_handle(self):
        """Nasty hack to set properties that legend expects to exist"""
        self._artists[0]._facecolors2d = self._artists[0]._facecolor3d
        self._artists[0]._edgecolors2d = self._artists[0]._edgecolor3d
        return self._artists[0]
