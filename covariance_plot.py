import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from scipy.stats import chi2

from plot_tools.covariance_ellipse_plotly import CovarianceEllipses2D

app = dash.Dash(__name__)

mean = [0, 0]
cov = [[1, 0], [0, 1]]
ellipse_masses = [0.68, 0.99]
cov_ellipse = CovarianceEllipses2D(ellipse_masses=ellipse_masses)

cov_Q = np.eye(2)
det_Q = np.linalg.det(cov_Q)
inv_covQ = np.linalg.inv(cov_Q)

# dcc.Markdown("Marginal ($x$) distribution $p(x)$", mathjax=True)
# dcc.Markdown("Marginal ($y$) distribution $p(y)$", mathjax=True)

app.layout = html.Div(
    [
        dcc.Markdown(
            "# Joint distribution properties (covariance ellipses) in 2D $p(x,y)$",
            mathjax=True,
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown(r"$\sigma_{xx}$", mathjax=True),
                        dcc.Slider(
                            id="sig_xx",
                            min=0,
                            max=3.0,
                            step=0.1,
                            value=1,
                            marks={i: str(i) for i in range(4)},
                        ),
                        dcc.Markdown(r"$\sigma_{yy}$", mathjax=True),
                        dcc.Slider(
                            id="sig_yy",
                            min=0,
                            max=3.0,
                            step=0.1,
                            value=1,
                            marks={i: str(i) for i in range(4)},
                        ),
                        dcc.Markdown(r"$\sigma_{xy}$", mathjax=True),
                        dcc.Slider(
                            id="sig_xy",
                            min=-1.0,
                            max=1.0,
                            step=0.05,
                            value=0,
                            marks={-1: "-1", -0.5: "-0.5", 0: "0", 0.5: "0.5", 1: "1"},
                        ),
                        html.Div(id="cov_matrix"),
                        html.Div(id="determinant"),
                        html.Div(id="trace"),
                        html.Div(id="min_eig"),
                        html.Div(id="kld"),
                    ],
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                        "width": "20%",
                    },
                ),
                dcc.Graph(id="graph_p_xy"),
                html.Div(
                    [
                        dcc.Graph(id="graph_p_x"),
                        dcc.Graph(id="graph_p_y"),
                    ],
                    style={"display": "flex", "flex-direction": "column"},
                ),
            ],
            style={"display": "flex", "flex-direction": "row"},
        ),
    ]
)

joint_layout = go.Layout(
    title=dict(text=r"Joint distribution $p(x,y)$"),
    xaxis=dict(title="x", range=[-10, 10], scaleanchor="y", scaleratio=1),
    yaxis=dict(title="y", range=[-10, 10]),
)
p_x_layout = go.Layout(
    title=dict(text=r"Marginal distribution $p(x)$"),
    xaxis=dict(title="x", range=[-5, 5]),
    yaxis=dict(title="p(x)", range=[0, 2]),
)
p_y_layout = go.Layout(
    title=dict(text=r"Marginal distribution $p(y)$"),
    xaxis=dict(title="y", range=[-5, 5]),
    yaxis=dict(title="p(y)", range=[0, 2]),
)


@app.callback(
    Output("graph_p_xy", "figure"),
    Output("graph_p_x", "figure"),
    Output("graph_p_y", "figure"),
    Output("sig_xy", "min"),
    Output("sig_xy", "max"),
    Output("sig_xy", "value"),
    Output("cov_matrix", "children"),
    Output("determinant", "children"),
    Output("trace", "children"),
    Output("min_eig", "children"),
    Output("kld", "children"),
    Input("sig_xx", "value"),
    Input("sig_yy", "value"),
    Input("sig_xy", "value"),
)
def update_graph(sig_xx, sig_yy, sig_xy):
    max_sig_xy = (sig_xx * sig_yy) ** 0.5
    sig_xy = min(max(sig_xy, -max_sig_xy), max_sig_xy)
    cov_update = np.array([[sig_xx**2, sig_xy**2], [sig_xy**2, sig_yy**2]])

    # Update the covariance object
    cov_ellipse.update_mean_cov(mean, cov_update)

    # Update the joint distribution plot
    gobj = cov_ellipse.make_ellipses()

    # Update the marginal plots
    marginals = cov_ellipse.marginals()

    # Matrix, determinant, trace
    det_P = np.linalg.det(cov_update)

    cov_matrix = dcc.Markdown(
        f"Covariance matrix $\\Sigma = \\begin{{bmatrix}} {cov_update[0,0]:0.2f} & {cov_update[0,1]:0.2f} \\\\  {cov_update[1,0]:0.2f} & {cov_update[1,1]:0.2f} \\end{{bmatrix}}$",
        mathjax=True,
    )
    determinant = dcc.Markdown(
        f"D-optimality: Determinant $\\det( \\Sigma ) = {det_P:0.2f}$",
        mathjax=True,
    )
    trace = dcc.Markdown(
        f"A-optimality: Trace $\\mathrm{{tr}}( \\Sigma ) = {np.trace(cov_update):0.2f}$",
        mathjax=True,
    )

    eigenvals = np.linalg.eigvalsh(cov_update)
    min_eig = dcc.Markdown(
        f"E-optimality: Min. eigenvalue $\\lambda_{{min}} = {eigenvals.min():0.2f}$",
        mathjax=True,
    )

    dkl = 0.5 * (np.log(det_Q / det_P) - 2 + np.trace(inv_covQ @ cov_update))
    kld = dcc.Markdown(
        f"KL-Divergence $D_{{\\mathrm{{KL}}}}( P || I_2 ) = {dkl:0.2f}$", mathjax=True
    )

    return (
        {"data": gobj, "layout": joint_layout},
        {"data": [marginals[0]], "layout": p_x_layout},
        {"data": [marginals[1]], "layout": p_y_layout},
        -max_sig_xy,
        max_sig_xy,
        sig_xy,
        cov_matrix,
        determinant,
        trace,
        min_eig,
        kld,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
