import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

#from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import column, layout, gridplot
from bokeh.models import Div, WheelZoomTool
#from bokeh.models.widgets import Panel, Tabs


def visualize_paths(gt_path, pred_path, html_tile="", title="VO exercises", file_out="plot.html"):
    output_file(file_out, title=html_tile)
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset"

    gt_x, gt_y = gt_path.T
    pred_x, pred_y = pred_path.T
    xs = list(np.array([gt_x, pred_x]).T)
    ys = list(np.array([gt_y, pred_y]).T)

    diff = np.linalg.norm(gt_path - pred_path, axis=1)
    source = ColumnDataSource(data=dict(gtx=gt_path[:, 0], gty=gt_path[:, 1],
                                        px=pred_path[:, 0], py=pred_path[:, 1],
                                        diffx=np.arange(len(diff)), diffy=diff,
                                        disx=xs, disy=ys))

    fig1 = figure(title="Paths", tools=tools, match_aspect=True, width_policy="max", toolbar_location="above",
                  x_axis_label="x", y_axis_label="y")
    fig1.circle("gtx", "gty", source=source, color="blue", hover_fill_color="firebrick", legend_label="GT")
    fig1.line("gtx", "gty", source=source, color="blue", legend_label="GT")

    fig1.circle("px", "py", source=source, color="green", hover_fill_color="firebrick", legend_label="Pred")
    fig1.line("px", "py", source=source, color="green", legend_label="Pred")

    fig1.multi_line("disx", "disy", source=source, legend_label="Error", color="red", line_dash="dashed")
    fig1.legend.click_policy = "hide"

    fig2 = figure(title="Error", tools=tools, width_policy="max", toolbar_location="above",
                  x_axis_label="frame", y_axis_label="error")
    fig2.circle("diffx", "diffy", source=source, hover_fill_color="firebrick", legend_label="Error")
    fig2.line("diffx", "diffy", source=source, legend_label="Error")

    show(layout([Div(text=f"<h1>{title}</h1>"),
                 Div(text="<h2>Paths</h1>"),
                 [fig1, fig2],
                 ], sizing_mode='scale_width'))


def make_residual_plot(x, residual_init, residual_minimized):
    fig1 = figure(title="Initial residuals", x_range=[0, len(residual_init)], x_axis_label="residual", y_axis_label="")
    fig1.line(x, residual_init)

    change = np.abs(residual_minimized) - np.abs(residual_init)
    plot_data = ColumnDataSource(data={"x": x, "residual": residual_minimized, "change": change})
    tooltips = [
        ("change", "@change"),
    ]
    fig2 = figure(title="Optimized residuals", x_axis_label=fig1.xaxis.axis_label, y_axis_label=fig1.yaxis.axis_label,
                  x_range=fig1.x_range, y_range=fig1.y_range, tooltips=tooltips)
    fig2.line("x", "residual", source=plot_data)

    fig3 = figure(title="Change", x_axis_label=fig1.xaxis.axis_label, y_axis_label=fig1.yaxis.axis_label,
                  x_range=fig1.x_range, tooltips=tooltips)
    fig3.line("x", "change", source=plot_data)
    return fig1, fig2, fig3


def plot_residual_results(qs_small, small_residual_init, small_residual_minimized,
                          qs, residual_init, residual_minimized):
    output_file("plot.html", title="Bundle Adjustment")
    x = np.arange(2 * qs_small.shape[0])
    fig1, fig2, fig3 = make_residual_plot(x, small_residual_init, small_residual_minimized)

    x = np.arange(2 * qs.shape[0])
    fig4, fig5, fig6 = make_residual_plot(x, residual_init, residual_minimized)

    show(layout([Div(text="<h1>Bundle Adjustment exercises</h1>"),
                 Div(text="<h2>Bundle adjustment with reduced parameters</h1>"),
                 gridplot([[fig1, fig2, fig3]], toolbar_location='above'),
                 Div(text="<h2>Bundle adjustment with all parameters (with sparsity)</h1>"),
                 gridplot([[fig4, fig5, fig6]], toolbar_location='above')
                 ]))


def plot_sparsity(sparse_mat):
    fig, ax = plt.subplots(figsize=[20, 10])
    plt.title("Sparsity matrix")

    ax.spy(sparse_mat, aspect="auto", markersize=0.02)
    plt.xlabel("Parameters")
    plt.ylabel("Resudals")

    plt.show()
