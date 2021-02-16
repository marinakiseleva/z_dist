import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from estimate.get_data import *
from estimate.constants import *

from thex_data.data_consts import *
from mainmodel.helper_compute import *


def init_plot_settings():
    """
    Set defaults for all plots: font.
    """
    plt.rcParams["font.family"] = "Times New Roman"


def plot_performance(model, testdata_y, output_dir, results):
    """
    Visualize all performance metrics for model.
    """
    # Reset class counts and y to be that of the test set, so baselines are accurate
    a = testdata_y.groupby('transient_type').size()[
        'I, Ia, _ROOT, _SN, _W_UVOPT, Unspecified Ia']
    b = testdata_y.groupby('transient_type').size()[
        'CC, II, _ROOT, _SN, _W_UVOPT, Unspecified II']
    model.class_counts = {"Unspecified Ia": a,
                          "Unspecified II": b}

    model.results = results
    model.y = testdata_y
    model.dir = output_dir

    if not os.path.exists(model.dir):
        os.mkdir(model.dir)

    # Save results in pickle
    with open(model.dir + '/results.pickle', 'wb') as f:
        pickle.dump(model.results, f)
    with open(model.dir + '/y.pickle', 'wb') as f:
        pickle.dump(model.y, f)

    model.visualize_performance()


def get_model_mets(model, results):
    """
    Gets purity and completeness (and confidence intervals) per class based
    on model results
    :param model: THEx model
    :param results: Results from trial runs
    """
    n = len(model.class_labels)

    N = model.num_runs
    pc_per_trial = model.get_pc_per_trial(results)
    ps, cs = model.get_avg_pc(pc_per_trial, N)

    p_intvls, c_intvls = compute_confintvls(pc_per_trial, model.class_labels)

    ps, cs = model.get_avg_pc(pc_per_trial, N)
    return ps, cs, p_intvls, c_intvls


def plot_class_met(ax, indices, vals, cis, baselines, label, color):
    """
    Plot the metrics for this class, on given axis.
    :param indices: y-value indices of bars (in horizontal bar plot)
    :param vals: Dict from class name to value
    :param cis: Dict from class name to [] confidence intervals
    :param baselines: dict from class name to baseline
    :param label: label for these bars. 
    """

    # Convert all dicts to same order of lists
    classes = ['Unspecified Ia', 'Unspecified II']
    val_list = []
    cis_list = []
    baselines_list = []
    for c in classes:
        val_list.append(vals[c])
        cis_list.append(cis[c])
        baselines_list.append(baselines[c])

    errs = prep_err_bars(cis_list, val_list)

    bar_width = 0.1
    INTVL_COLOR = "black"
    BSLN_COLOR = "#ff1a1a"
    BAR_EDGE_COLOR = "black"
    ax.barh(y=indices,
            width=val_list,
            height=bar_width,
            xerr=errs,
            capsize=9,
            edgecolor=BAR_EDGE_COLOR,
            ecolor=INTVL_COLOR,
            color=color,
            label=label)
    for index, baseline in enumerate(baselines_list):
        y_val = indices[index]
        ax.vlines(x=baseline,
                  ymin=y_val - (bar_width / 2),
                  ymax=y_val + (bar_width / 2),
                  linewidth=3,
                  linestyles=(0, (1, 1)), colors=BSLN_COLOR)


def plot_met(axis, model, L_vals, L_cis, r_vals, r_cis, baselines, label):
    """
    Given LSST and random metrics (either purity or completeness) plot on figure
    """

    plot_class_met(ax=axis,
                   indices=[0.3, 0.5],
                   vals=r_vals,
                   cis=r_cis,
                   baselines=baselines,
                   label="THEx test\nsets",
                   color=THEX_COLOR)

    plot_class_met(ax=axis,
                   indices=[0.2, 0.4],
                   vals=L_vals,
                   cis=L_cis,
                   baselines=baselines,
                   label="LSST-like\ntest sets",
                   color=LSST_SAMPLE_COLOR)

    # Figure formatting
    axis.set_xlim(0, 1)
    indices = np.linspace(0, 1, 6)
    ticks = ['0', '20', '40', '60', '80', '']
    axis.set_xticks(indices)
    axis.set_xticklabels(ticks, fontsize=16)
    axis.set_xlabel(label + " (%)", fontsize=16)


def plot_performance_together(model, test_y, LSST_results, orig_results):
    """
    Plot performance of LSST test set vs regular test set; purity and completeness.
    :param model: Initialized THEx model used
    :param test_y: One of the test sets. Only use it for the class counts. Assumes both test sets have same class counts.
    :param LSST_results: Results of LSST test data
    :param orig_results: Results of THEx test data
    """
    L_ps, L_cs, L_ps_ci, L_cs_ci = get_model_mets(model, LSST_results)
    r_ps, r_cs, r_ps_ci, r_cs_ci = get_model_mets(model, orig_results)

    c_baselines, p_baselines = compute_baselines(
        model.class_counts, model.class_labels, test_y, len(model.class_labels), None)

    fig, ax = plt.subplots(figsize=(6, 3),
                           nrows=1, ncols=2,
                           dpi=600,
                           sharex=True,
                           sharey=True)

    plot_met(axis=ax[0],
             model=model,
             L_vals=L_ps,
             L_cis=L_ps_ci,
             r_vals=r_ps,
             r_cis=r_ps_ci,
             baselines=p_baselines,
             label="Purity")

    plot_met(axis=ax[1],
             model=model,
             L_vals=L_cs,
             L_cis=L_cs_ci,
             r_vals=r_cs,
             r_cis=r_cs_ci,
             baselines=c_baselines,
             label="Completeness")

    # bbox_to_anchor=(1.1., 1, 0.3, .6),        (x, y, width, height)
    ax[1].legend(fontsize=16, bbox_to_anchor=(1.1, 0.7),
                 labelspacing=.2, handlelength=1)
    ax[0].set_yticks(np.array([0.3, 0.5]) - 0.05)
    new_labels = ["Ia (unspec.)", "II (unspec.)"]
    ax[0].set_yticklabels(new_labels,  fontsize=16, horizontalalignment='right')
    plt.gca().invert_yaxis()

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig("../figures/testing/LSST_Evaluation_Metrics.pdf", bbox_inches='tight')
