import dash
import numpy as np
import plotly.graph_objs as go
from dash import ctx, dcc, html
from dash.dependencies import Input, Output
from scipy.stats import chi2

from plot_tools.covariance_ellipse_plotly import CovarianceEllipses2D

app = dash.Dash(__name__)

mean = [0, 0]
cov = np.array([[1, 0], [0, 1]])
ellipse_masses = [0.68, 0.99]
cov_ellipse = CovarianceEllipses2D(ellipse_masses=ellipse_masses)
pts = []

cov_Q = np.eye(2)
det_Q = np.linalg.det(cov_Q)
inv_covQ = np.linalg.inv(cov_Q)

n_resets = 0

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

app.layout = html.Div(
    [
        dcc.Markdown(
            r"# Joint distribution properties (covariance ellipses) in 2D $p(x,y) = \mathcal{N}(\mu,\Sigma)$",
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
                        dcc.Markdown(r"$\mathrm{cov}_{xy}$", mathjax=True),
                        dcc.Slider(
                            id="cov_xy",
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
                        html.Button('Reset', id='reset_button', n_clicks=0),
                    ],
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                        "width": "20%",
                    },
                ),
                dcc.Graph(id="graph_p_xy", figure={'layout': joint_layout}), #, layout={'clickmode': 'event+select'}),
                # html.Div(id="click-output"),
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


@app.callback(
    Output("graph_p_xy", "figure"),
    # Output("click-output", "children"),
    Output("graph_p_x", "figure"),
    Output("graph_p_y", "figure"),
    Output("sig_xx", "value"),
    Output("sig_yy", "value"),
    Output("cov_xy", "min"),
    Output("cov_xy", "max"),
    Output("cov_xy", "value"),
    Output("cov_matrix", "children"),
    Output("determinant", "children"),
    Output("trace", "children"),
    Output("min_eig", "children"),
    Output("kld", "children"),
    Input("sig_xx", "value"),
    Input("sig_yy", "value"),
    Input("cov_xy", "value"),
    Input("reset_button", "n_clicks")
    # Input("graph_p_xy", "clickData"),
)
def update_graph(sig_xx, sig_yy, cov_xy, n_clicks_reset):
    if ctx.triggered_id == 'reset_button':
        sig_xx = cov[0,0]
        sig_yy = cov[1,1]
        cov_xy = cov[0,1]

    max_cov_xy = sig_xx * sig_yy
    cov_xy = min(max(cov_xy, -max_cov_xy), max_cov_xy)
    cov_update = np.array([[sig_xx**2, cov_xy], [cov_xy, sig_yy**2]])

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
        f"D-optimality: Determinant $\\det( \\Sigma^{{-1}} ) = {1.0/det_P:0.2f}$",
        mathjax=True,
    )
    trace = dcc.Markdown(
        f"A-optimality: Trace $\\mathrm{{tr}}( \\Sigma ) = {np.trace(cov_update):0.2f}$",
        mathjax=True,
    )

    eigenvals = np.linalg.eigvalsh(cov_update)
    min_eig = dcc.Markdown(
        f"E-optimality: Min. eigenvalue $\\lambda_{{min}} = {1.0/eigenvals.max():0.2f}$",
        mathjax=True,
    )

    dkl = 0.5 * (np.log(det_Q / det_P) - 2 + np.trace(inv_covQ @ cov_update))
    kld = dcc.Markdown(
        f"KL-Divergence $D_{{\\mathrm{{KL}}}}( P || I_2 ) = {dkl:0.2f}$", mathjax=True
    )

    # # If clicked for observation
    # if clickData:
    #     pt = clickData["points"][0]
    #     # cov_ellipse.add_observation(pt)
    #     pts.append(pt)
    #     gobj.append(
    #         go.Scatter(
    #             x=[p[0] for p in pts],
    #             y=[p[1] for p in pts],
    #             mode="points",
    #             name="observations",
    #         )
    #     )

    return (
        {"data": gobj, "layout": joint_layout},
        {"data": [marginals[0]], "layout": p_x_layout},
        {"data": [marginals[1]], "layout": p_y_layout},
        sig_xx,
        sig_yy,
        -max_cov_xy,
        max_cov_xy,
        cov_xy,
        cov_matrix,
        determinant,
        trace,
        min_eig,
        kld,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
