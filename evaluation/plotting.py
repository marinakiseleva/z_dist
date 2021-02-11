import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from estimate.get_data import *
from thex_data.data_consts import *
from mainmodel.helper_compute import *
from matplotlib import rc


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
            capsize=7,
            edgecolor=BAR_EDGE_COLOR,
            ecolor=INTVL_COLOR,
            color=color,
            label=label)
    for index, baseline in enumerate(baselines_list):
        y_val = indices[index]
        plt.vlines(x=baseline,
                   ymin=y_val - (bar_width / 2),
                   ymax=y_val + (bar_width / 2),
                   linewidth=3,
                   linestyles=(0, (1, 1)), colors=BSLN_COLOR)


def plot_met(model, L_vals, L_cis, r_vals, r_cis, baselines, label):
    """
    Given LSST and random metrics (either purity or completeness) plot on figure
    """
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                           dpi=150,
                           sharex=True,
                           sharey=True,
                           tight_layout=True)

    THEX_COLOR = "#ffa31a"
    LSST_SAMPLE_COLOR = "#24248f"

    plot_class_met(ax=ax,
                   indices=[0.3, 0.5],
                   vals=r_vals,
                   cis=r_cis,
                   baselines=baselines,
                   label="THEx test sets",
                   color=THEX_COLOR)

    plot_class_met(ax=ax,
                   indices=[0.2, 0.4],
                   vals=L_vals,
                   cis=L_cis,
                   baselines=baselines,
                   label="LSST-like test sets",
                   color=LSST_SAMPLE_COLOR)

    if label == 'Completeness':
        ax.legend(loc='lower right', fontsize=14)
    # Figure formatting
    ax.set_xlim(0, 1)
    indices = np.linspace(0, 1, 6)
    ticks = [str(int(i)) for i in indices * 100]
    plt.xticks(indices, ticks, fontsize=16)

    plt.xlabel(label + " (%)", fontsize=20)
    new_labels = ["Ia (unspec.)", "II (unspec.)"]
    plt.yticks(np.array([0.3, 0.5]) - 0.05, new_labels,  fontsize=20,
               horizontalalignment='right')

    plt.gca().invert_yaxis()
    plt.savefig("../figures/testing/LSST_Evaluation_" + label + ".pdf")


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

    plot_met(model=model,
             L_vals=L_ps,
             L_cis=L_ps_ci,
             r_vals=r_ps,
             r_cis=r_ps_ci,
             baselines=p_baselines,
             label="Purity")

    plot_met(model=model,
             L_vals=L_cs,
             L_cis=L_cs_ci,
             r_vals=r_cs,
             r_cis=r_cs_ci,
             baselines=c_baselines,
             label="Completeness")
