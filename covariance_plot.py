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

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown(
                            "Joint distribution (covariance ellipse) $p(x,y)$",
                            mathjax=True,
                        ),
                        dcc.Graph(id="graph_p_xy"),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "Marginal ($x$) distribution $p(x)$", mathjax=True
                        ),
                        dcc.Graph(id="graph_p_x"),
                    ]
                ),
                html.Div(
                    [
                        dcc.Markdown(
                            "Marginal ($y$) distribution $p(y)$", mathjax=True
                        ),
                        dcc.Graph(id="graph_p_y"),
                    ]
                ),
            ],
            style={"display": "flex", "flex-direction": "row"},
        ),
        dcc.Markdown(r"$\sigma_{xx}$", mathjax=True),
        # html.Label("σₓₓ"),
        dcc.Slider(
            id="sig_xx",
            min=0,
            max=5.0,
            step=0.1,
            value=1,
            marks={i: str(i) for i in range(6)},
        ),
        dcc.Markdown(r"$\sigma_{yy}$", mathjax=True),
        dcc.Slider(
            id="sig_yy",
            min=0,
            max=5.0,
            step=0.1,
            value=1,
            marks={i: str(i) for i in range(6)},
        ),
        dcc.Markdown(r"$\sigma_{xy}$", mathjax=True),
        dcc.Slider(
            id="sig_xy",
            min=-1.0,
            max=1.0,
            step=0.05,
            value=0,
            marks={-1: "-1", 0: "0", 1: "1"},
        ),
    ]
)

joint_layout = go.Layout(
    xaxis=dict(title="x", range=[-10, 10], scaleanchor="y", scaleratio=1),
    yaxis=dict(title="y", range=[-10, 10]),
)
p_x_layout = go.Layout(
    xaxis=dict(title="x", range=[-5, 5]),
    yaxis=dict(title="p(x)", range=[0, 2]),
)
p_y_layout = go.Layout(
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
    Input("sig_xx", "value"),
    Input("sig_yy", "value"),
    Input("sig_xy", "value"),
)
def update_graph(sig_xx, sig_yy, sig_xy):
    max_sig_xy = (sig_xx * sig_yy) ** 0.5
    sig_xy = min(max(sig_xy, -max_sig_xy), max_sig_xy)
    cov_update = [[sig_xx**2, sig_xy**2], [sig_xy**2, sig_yy**2]]

    # Update the covariance object
    cov_ellipse.update_mean_cov(mean, cov_update)

    # Update the joint distribution plot
    gobj = cov_ellipse.make_ellipses()

    # Update the marginal plots
    marginals = cov_ellipse.marginals()

    return (
        {"data": gobj, "layout": joint_layout},
        {"data": [marginals[0]], "layout": p_x_layout},
        {"data": [marginals[1]], "layout": p_y_layout},
        -max_sig_xy,
        max_sig_xy,
        sig_xy,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
