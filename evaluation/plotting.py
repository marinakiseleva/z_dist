import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from estimate.get_data import *
from thex_data.data_consts import *
from mainmodel.helper_compute import *


def init_plot_settings():
    """
    Set defaults for all plots: font.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def plot_compare_z_dists(class_name, thex_dataset, output_dir):
    """
    Compare overall z distributions of THEx data versus LSST data
    """

    thex_class_data = get_thex_class_data(class_name, thex_dataset)

    data = get_lsst_data()
    lsst_class_name = class_name
    pretty_class_name = class_name
    if 'Unspecified ' in class_name:
        lsst_class_name = class_name.replace('Unspecified ', "")
        pretty_class_name = lsst_class_name + " (unspec.)"
    lsst_class_data = data[lsst_class_name]
    feature_data = lsst_class_data['true_z']
    indices = []
    for index, f in enumerate(feature_data):
        if ~np.isnan(f):
            indices.append(index)

    valid_Z = np.take(lsst_class_data['true_z'], indices)
    lsst_class_data = valid_Z.tolist()

    lsst_label = "Rubin"

    Z_bins = np.linspace(0, 1, 50)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                           dpi=DPI,
                           tight_layout=True, sharex=True,  sharey=True)

    GREEN = "#b3e6b3"
    BLUE = "#99c2ff"
    RED = "#ffb3b3"

    a = ax.hist(lsst_class_data, density=True, bins=Z_bins,
                label=lsst_label, fill=False, edgecolor=BLUE, linewidth=1.2)
    b = ax.hist(thex_class_data['redshift'].values, density=True, bins=Z_bins,
                label="THEx", fill=False, edgecolor=GREEN, linewidth=1.2)
    plt.xticks(fontsize=TICK_S)
    plt.yticks(fontsize=TICK_S)
    plt.legend(fontsize=LAB_S)

    plt.title(pretty_class_name, fontsize=TITLE_S)
    plt.xlabel("Redshift", fontsize=LAB_S)
    plt.ylabel("Density", fontsize=LAB_S)
    plt.savefig(output_dir + "/" + pretty_class_name + "_overall.pdf")


def plot_compare_feature_dists(feature_name, class_name, rand_sample, sampled):
    FIG_WIDTH = 6
    FIG_HEIGHT = 4
    DPI = 200
    GREEN = "#b3e6b3"
    BLUE = "#99c2ff"
    RED = "#ffb3b3"

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI,
                           tight_layout=True, sharex=True,  sharey=True)

    if feature_name == 'redshift':
        bins = np.linspace(0, 1, 20)
        plt.xlim((0, 1))
    else:
        # mag
        bins = np.linspace(9, 23, 50)
        plt.xlim((9, 23))

    b = ax.hist(rand_sample[feature_name].values, bins=bins, density=True,
                label="THEx (random sample)", fill=False, edgecolor=GREEN)
    a = ax.hist(sampled[feature_name].values, bins=bins, density=True,
                label="THEx (LSST sample)", fill=False, edgecolor=RED)

    plt.legend(fontsize=12)
    plt.title(class_name, fontsize=14)
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.savefig("../figures/evaluation/feature_dist_" +
                feature_name + "_" + class_name + ".pdf")


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
    ax.barh(y=indices,
            width=val_list,
            height=bar_width,
            xerr=errs,
            capsize=2, edgecolor='black', ecolor='coral', color=color,
            label=label)
    for index, baseline in enumerate(baselines_list):
        y_val = indices[index]
        plt.vlines(x=baseline,
                   ymin=y_val - (bar_width / 2),
                   ymax=y_val + (bar_width / 2),
                   linestyles='--', colors='red')


def plot_met(model, L_vals, L_cis, r_vals, r_cis, baselines, label):
    """
    Given LSST and random metrics (either purity or completeness) plot on figure
    """

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=200,
                           tight_layout=True, sharex=True, sharey=True)
    GREEN = "#b3e6b3"
    RED = "#ffb3b3"
    plot_class_met(ax=ax,
                   indices=[0.3, 0.6],
                   vals=r_vals,
                   cis=r_cis,
                   baselines=baselines,
                   label="THEx",
                   color=GREEN)

    plot_class_met(ax=ax,
                   indices=[0.2, 0.5],
                   vals=L_vals,
                   cis=L_cis,
                   baselines=baselines,
                   label="LSST-like",
                   color=RED)

    # Figure formatting
    ax.set_xlim(0, 1)
    x_indices, x_ticks = get_perc_ticks()
    plt.xticks(x_indices, x_ticks, fontsize=TICK_S)

    plt.legend(loc='best', fontsize=LAB_S)
    plt.yticks(np.array([0.3, 0.6]) - 0.05, model.class_labels,  fontsize=TICK_S,
               horizontalalignment='right')
    plt.ylabel('Transient Class', fontsize=LAB_S)
    plt.xlabel(label, fontsize=LAB_S)
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
