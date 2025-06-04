import numpy as np
import plotly.graph_objs as go
from scipy.stats import chi2
from plot_tools.covariance_ellipse import _check_mean_cov


class CovarianceEllipses2D:
    """Handler for plotting and animating 2D covariance ellipses in plotly"""

    def __init__(
        self,
        ellipse_masses: list | np.ndarray = [0.68, 0.95],
        n_theta: int = 100,
        n_marginals: int = 50,
    ) -> None:
        self._ellipse_scales = np.sqrt(chi2.ppf(ellipse_masses, df=2))
        self.theta = np.linspace(0, 2 * np.pi, n_theta)
        self.m_x = np.linspace(-5, 5, n_marginals)

    def update_mean_cov(self, mean: list | np.ndarray, cov: list | np.ndarray):
        self.mean, self.cov = _check_mean_cov(mean, cov)

    def make_ellipses(self) -> list[go.Scatter]:
        d1, d2, angle = self._get_ellipse_params()
        graph_obj = []
        for s in self._ellipse_scales:
            x = (
                s * d1 * np.cos(self.theta) * np.cos(angle)
                - (s * d2 * np.sin(self.theta) * np.sin(angle))
                + self.mean[0]
            )
            y = (
                s * d1 * np.cos(self.theta) * np.sin(angle)
                + (s * d2 * np.sin(self.theta) * np.cos(angle))
                + self.mean[1]
            )
            graph_obj.append(go.Scatter(x=x, y=y, mode="lines"))
        return graph_obj

    def marginals(self) -> list[go.Scatter]:
        graph_obj = []
        for sig in [self.cov[0, 0], self.cov[1, 1]]:
            p = np.exp(-0.5 * (self.m_x / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
            graph_obj.append(go.Scatter(x=self.m_x, y=p, mode="lines"))
        return graph_obj

    def _get_ellipse_params(self):
        # Get ellipse parameters (axes and angle in rad)
        # Get eigenvalues and sort them (big first)
        w, V = np.linalg.eig(self.cov)
        idx = -w.argsort()[::-1]
        w = w[idx]
        V = V[:, idx]
        d1, d2 = 2 * np.sqrt(w)
        angle = np.arctan2(V[1, 0], V[0, 0])
        return d1, d2, angle
