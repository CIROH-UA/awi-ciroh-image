""" Plotting logic for TEVA """
from typing import Callable
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

from teva.ccea import conjunctive_clause
from teva.dnfea import disjunctive_clause
from teva.utilities import hyge

matplotlib.rcParams.update({'font.size': 5})

def _color_lerp(start: tuple, end: tuple, ratio: float) -> tuple:
    lerp_r: float = (start[0] + (end[0] - start[0]) * ratio)
    lerp_g: float = (start[1] + (end[1] - start[1]) * ratio)
    lerp_b: float = (start[2] + (end[2] - start[2]) * ratio)

    return lerp_r, lerp_g, lerp_b

def _plot_cc_order(conjunctive_clauses: list[conjunctive_clause.ConjunctiveClause],
                   order: int,
                   axis: plt.axis,
                   color: tuple,
                   point_size: float = 8,
                   marker: str = "o"):
    """ Plots the CCs of a single clause order onto the given axis

    :param conjunctive_clauses: The list of ccs to plot
    :param order: The order of cc clauses to plot
    :param axis: The axis on which to plot the ccs
    :param color: The color the ccs of this order should be plotted as
    """
    coverage = [clause.cov for clause in conjunctive_clauses]
    ppv = [clause.ppv for clause in conjunctive_clauses]

    if axis is None:
        axis = plt

    axis.scatter(x=coverage, y=ppv,
                 marker=marker,
                 label=f"CC Order: {order}",
                 zorder=5,
                 s=point_size,
                 edgecolors='none',
                 color=color)

def plot_ccs(conjunctive_clauses: dict[int: list[conjunctive_clause.ConjunctiveClause]],
             axis: plt.axis = None,
             point_size: float = 8,
             marker: str = "o",
             min_order_color: tuple = (0.75, 1.0, 0.75),
             max_order_color: tuple = (0.0, 1.0, 0.0)):
    """ Plots all provided CCs, coloring them separately based on clause order

    :param conjunctive_clauses: The dictionary of CC clause orders to plot
    :param axis: The axis on which to plot
    :param marker: The marker style for the CC points
    :param point_size: The size of the CC points
    :param min_order_color: The color of the minimum order
    :param max_order_color: The color of the maximum order
    """
    max_order = max(conjunctive_clauses.keys())

    for order in sorted(conjunctive_clauses.keys()):
        _plot_cc_order(conjunctive_clauses[order],
                       order,
                       axis,
                       color=_color_lerp(min_order_color, max_order_color, (order - 1)/max_order),
                       point_size=point_size,
                       marker=marker)
    axis.legend(loc="lower left")

def _plot_dnf_order(disjunctive_clauses: list[disjunctive_clause.DisjunctiveClause],
                    order: int,
                    axis: plt.axis,
                    color: tuple,
                    point_size: float = 8,
                    marker: str = "s"):
    """ Plots the DNFs of a single clause order onto the given axis

    :param disjunctive_clauses: The list of dnfs to plot
    :param order: The order of dnf clauses to plot
    :param axis: The axis on which to plot the dnfs
    :param color: The color the dnfs of this order should be plotted as
    """
    coverage = [clause.cov for clause in disjunctive_clauses]
    ppv = [clause.ppv for clause in disjunctive_clauses]

    if axis is None:
        axis = plt

    axis.scatter(x=coverage, y=ppv,
                 marker=marker,
                 label=f"DNF Order: {order}",
                 zorder=5,
                 s=point_size,
                 edgecolors='none',
                 color=color)

def plot_dnfs(disjunctive_clauses: dict[int: list[disjunctive_clause.DisjunctiveClause]],
              axis: plt.axis = None,
              point_size: float = 8,
              marker: str = "s",
              min_order_color: tuple = (0.75, 0.75, 1.0),
              max_order_color: tuple = (0.0, 0.0, 1.0),):
    """ Plots all provided DNFs, coloring them separately based on clause order

    :param disjunctive_clauses: The dictionary of DNF clause orders to plot
    :param axis: The axis on which to plot
    :param marker: The marker style for the DNF points
    :param point_size: The size of the DNF points
    :param min_order_color: The color of the minimum order
    :param max_order_color: The color of the maximum order
    """
    max_order = max(disjunctive_clauses.keys())

    for order in sorted(disjunctive_clauses.keys()):
        _plot_dnf_order(disjunctive_clauses[order],
                        order,
                        axis,
                        color=_color_lerp(min_order_color, max_order_color, (order - 1)/max_order),
                        point_size=point_size,
                        marker=marker)
    axis.legend(loc="lower right")

def get_contours(classes: np.ndarray,
                 target_class,
                 fit_interval: int = 25,
                 fitness_function: Callable = hyge.hygepmf) -> (np.ndarray, np.ndarray):
    """ Returns the fitness contours of the given target class

    :param classes: All unique classes from the input data
    :param target_class: The targeted class
    :param fit_interval: The interval of contour fitness lines
    :param fitness_function: The fitness function to use (default: :function:`hygepmf()`)

    :return: An array of fitness levels, an array of contour boundaries
    """
    return hyge.hygcontours(classes, target_class, fit_interval, fitness_function)

def plot_contours(fitness_levels,
                  boundaries: list,
                  axis: plt.axis = None,
                  color: tuple = (0.75, 0.75, 0.75),
                  linestyle="--",
                  linewidth = 0.5):
    """ Plots contour lines onto a specified axis

    :param fitness_levels: The fitness levels generated by :function:`get_contours()`
    :param boundaries: The boundaries generated by :function:`get_contours()`
    :param axis: The axis on which to plot the contours

    :param color: The color of the countour lines
    :param linestyle: The style of the contour lines
    :param linewidth: The width of the contour lines
    """
    if len(boundaries) == 0:
        print("No valid boundaries to plot.")

    if axis is None:
        _, axis = plt.subplots(1)

    for i in range(min(fitness_levels.shape[0], len(boundaries))):
        cur_cov = np.array(boundaries[i]["bound_coverage"])
        cur_ppv = np.array(boundaries[i]["mean_bound_ppv"])
        # NOTE: Why is it a step of 8?
        plot_cov = np.hstack((cur_cov[np.arange(0, cur_cov.shape[0], 8)], cur_cov[-1]))
        plot_ppv = np.hstack((cur_ppv[np.arange(0, cur_ppv.shape[0], 8)], cur_ppv[-1]))

        axis.plot(plot_cov, plot_ppv,
                  c=color,
                  linestyle=linestyle,
                  zorder=0,
                  linewidth=linewidth)

def plot_all(conjunctive_clauses: dict[int: list[conjunctive_clause.ConjunctiveClause]],
             disjunctive_clauses: dict[int: list[disjunctive_clause.DisjunctiveClause]],
             classes,
             target_class,

             x_label: str = "Observation Coverage",
             autoscale_x_axis: bool = False,
             autoscale_y_axis: bool = False,

             contour_color: tuple = (0.75, 0.75, 0.75),
             contour_linestyle="--",
             contour_linewidth = 0.5,

             cc_point_size: float = 8,
             cc_marker: str = "o",
             cc_min_order_color: tuple = (0.75, 1.0, 0.75),
             cc_max_order_color: tuple = (0.0, 1.0, 0.0),
             cc_ylabel: str = "CC Positive Predictive Value",

             dnf_point_size: float = 8,
             dnf_marker: str = "s",
             dnf_min_order_color: tuple = (0.75, 0.75, 1.0),
             dnf_max_order_color: tuple = (0.0, 0.0, 1.0),
             dnf_ylabel: str = "DNF Positive Predictive Value"):
    """ Plots all the given clauses, as well as the contour lines for the given target class

    :param conjunctive_clauses: A dictionary of CC orders to plot
    :param disjunctive_clauses: A dictionary of DNF orders to plot
    :param classes: A list of all unique classes
    :param target_class: The target class
    :param x_label: The label of the X axis
    :param autoscale_x_axis: If True, the x-axis will be autoscaled
    :param autoscale_y_axis: If True, the y-axis will be autoscaled

    :param contour_color: The color of the countour lines
    :param contour_linestyle: The style of the contour lines
    :param contour_linewidth: The width of the contour lines

    :param cc_point_size: The marker style for the CC points
    :param cc_marker: The size of the CC points
    :param cc_min_order_color: The color of the minimum CC order
    :param cc_max_order_color: The color of the maximum CC order
    :param cc_ylabel: The label of the Y axis for CCs

    :param dnf_point_size: The marker style for the DNF points
    :param dnf_marker: The size of the DNF points
    :param dnf_min_order_color: The color of the minimum DNF order
    :param dnf_max_order_color: The color of the maximum DNF order
    :param dnf_ylabel: The label of the Y axis for DNFs
    """
    _, ax = plt.subplots(1, dpi=300)

    fitnesses, boundaries = get_contours(classes, target_class)

    plot_contours(fitnesses,
                  boundaries,
                  axis=ax,
                  color=contour_color,
                  linestyle=contour_linestyle,
                  linewidth=contour_linewidth)

    plot_ccs(conjunctive_clauses,
             ax,
             point_size=cc_point_size,
             marker=cc_marker,
             min_order_color=cc_min_order_color,
             max_order_color=cc_max_order_color)

    ax2 = ax.twinx()

    plot_dnfs(disjunctive_clauses,
              ax2,
              point_size=dnf_point_size,
              marker=dnf_marker,
              min_order_color=dnf_min_order_color,
              max_order_color=dnf_max_order_color)

    if not autoscale_x_axis:
        ax.set_xlim(0.0, 1.05)
        ax2.set_xlim(0.0, 1.05)

    if not autoscale_y_axis:
        ax.set_ylim(0.0, 1.05)
        ax2.set_ylim(0.0, 1.05)

    ax.set_xlabel(x_label)

    ax.set_ylabel(cc_ylabel)
    ax2.set_ylabel(dnf_ylabel)

    ax.grid(True, linewidth=0.25)
    ax2.grid(False)

    plt.show()
