import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl

from thex_data.data_consts import *
import utilities.utilities as thex_utils
from mainmodel.helper_plotting import *

from estimate.get_data import *
from estimate.constants import *


def init_plot_settings():
    """
    Set defaults for all plots: font.
    """
    rc('text', usetex=True)
    mpl.rcParams['font.serif'] = ['times', 'times new roman']
    mpl.rcParams['font.family'] = 'serif'


def plot_sample_dist(ax, rand_sample, lsst_sample, lsst_orig, class_name):

    THEX_COLOR = "#ffa31a"
    LSST_COLOR = "#80ccff"
    LSST_SAMPLE_COLOR = "#24248f"

    def plot_step(data, bins, axis, label, color):
        """
        Get hist data and plot as step graph, no inner lines
        """
        # bins values in bin are (for left, right), left <= x < right
        vals, bins = np.histogram(data, bins=bins, density=True)
        a = np.array([0])
        bin_indices = np.linspace(min(bins), max(bins), len(bins))
        bin_indices = bin_indices[1:]
        xnew = np.concatenate((a, bin_indices), axis=0)
        ynew = np.concatenate((a, vals), axis=0)

        axis.step(x=xnew,
                  y=ynew,
                  label=label,
                  linewidth=2,
                  color=color)

    Z_bins = np.linspace(0, 1, 50)

    a = ax.hist(lsst_orig,
                density=True,
                bins=Z_bins,
                label="LSST",
                fill=True,
                alpha=0.8,
                color=LSST_COLOR)

    plot_step(data=rand_sample[Z_FEAT].values,
              bins=Z_bins,
              axis=ax,
              label="THEx test set",
              color=THEX_COLOR)
    plot_step(data=lsst_sample[Z_FEAT].values,
              bins=Z_bins,
              axis=ax,
              label="LSST-like test set",
              color=LSST_SAMPLE_COLOR)

    ax.set_title(class_name, fontsize=22, y=0.8, x=0.75)


def plot_sample_dists_together(Ia_sampled, Ia_rand_sample, Ia_LSST_Z, II_sampled, II_rand_sample, II_LSST_Z, output_dir):
    """
    Plot LSST orig vs THEx sample vs LSST sample for each class on shared fig.
    """
    # Plot LSST data, sampled LSST, and random sample
    rc('font', family="Times New Roman")
    f, ax = plt.subplots(nrows=2,
                         ncols=1,
                         sharex=True, sharey=True,
                         figsize=(5, 7),
                         dpi=140)

    plot_sample_dist(ax[0], Ia_rand_sample, Ia_sampled, Ia_LSST_Z, "Ia (unspec.)")
    plot_sample_dist(ax[1], II_rand_sample, II_sampled, II_LSST_Z, "II")

    ax[0].legend(fontsize=14, loc="upper left",  labelspacing=.2, handlelength=1)

    ax[0].yaxis.set_tick_params(labelsize=16)
    ax[1].yaxis.set_tick_params(labelsize=16)
    ax[1].xaxis.set_tick_params(labelsize=16)

    ax[1].set_xlabel("Redshift", fontsize=20)
    ax[0].set_ylabel("Density", fontsize=20)
    ax[1].set_ylabel("Density", fontsize=20)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_dir + "/samples.pdf", bbox_inches='tight')


def plot_performance(model, testdata_y, output_dir, results):
    """
    Visualize all performance metrics for model.
    """
    def get_class_count(df, cname):
        count = 0
        for i, row in df.iterrows():
            if cname in row[TARGET_LABEL]:
                count += 1
        return count
    # Reset class counts and y to be that of the test set, so baselines are accurate
    for c in model.class_labels:
        model.class_counts[c] = get_class_count(testdata_y, c)

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

    pc_per_trial = model.get_pc_per_trial(results)
    ps, cs = model.get_pc_performance(pc_per_trial)
    p_intvls, c_intvls = compute_confintvls(pc_per_trial, model.class_labels, False)

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
    classes = ['Unspecified Ia', 'II']
    val_list = []
    cis_list = []
    baselines_list = []
    for c in classes:
        val_list.append(vals[c])
        cis_list.append(cis[c])
        baselines_list.append(baselines[c])

    errs = prep_err_bars(cis_list, val_list)

    thex_utils.pretty_print_mets(
        class_labels=classes,
        vals=val_list,
        baselines=baselines_list,
        intvls=cis_list)

    bar_width = 0.05
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
        ax.vlines(x=baseline,
                  ymin=y_val - (bar_width / 2),
                  ymax=y_val + (bar_width / 2),
                  linewidth=2,
                  linestyles=(0, (1, 1)), colors=BSLN_COLOR)


def plot_met(axis, model, L_vals, L_cis, r_vals, r_cis, baselines, label):
    """
    Given LSST and random metrics (either purity or completeness) plot on figure
    """

    thex_y_points = [0.2, 0.3]
    lsst_y_points = [0.25, 0.35]

    print("\nTHEx Sample")
    plot_class_met(ax=axis,
                   indices=thex_y_points,
                   vals=r_vals,
                   cis=r_cis,
                   baselines=baselines,
                   label="THEx test\nsets",
                   color=THEX_COLOR)
    print("\nLSST Sample")
    plot_class_met(ax=axis,
                   indices=lsst_y_points,
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
    axis.set_xticklabels(ticks, fontsize=14)
    axis.set_xlabel(label + " (\%)", fontsize=14)


def plot_performance_together(model, test_y, LSST_results, orig_results, output_dir):
    """
    Plot performance of LSST test set vs regular test set; purity and completeness.
    :param model: Initialized THEx model used
    :param test_y: One of the test sets. Only use it for the class counts. Assumes both test sets have same class counts.
    :param LSST_results: Results of LSST test data
    :param orig_results: Results of THEx test data
    """
    L_ps, L_cs, L_ps_ci, L_cs_ci = get_model_mets(model, LSST_results)
    r_ps, r_cs, r_ps_ci, r_cs_ci = get_model_mets(model, orig_results)

    c_baselines, p_baselines = compute_baselines(class_counts=model.class_counts,
                                                 class_labels=model.class_labels,
                                                 N=model.get_num_classes(),
                                                 balanced_purity=model.balanced_purity,
                                                 class_priors=model.class_priors)

    fig, ax = plt.subplots(figsize=(4, 1.5),
                           nrows=1, ncols=2,
                           dpi=150,
                           sharex=True,
                           sharey=True)
    print("\n\n ****************** PURITY ***************")
    plot_met(axis=ax[0],
             model=model,
             L_vals=L_ps,
             L_cis=L_ps_ci,
             r_vals=r_ps,
             r_cis=r_ps_ci,
             baselines=p_baselines,
             label="Purity")

    print("\n\n ****************** COMPLETENESS ***************")
    plot_met(axis=ax[1],
             model=model,
             L_vals=L_cs,
             L_cis=L_cs_ci,
             r_vals=r_cs,
             r_cis=r_cs_ci,
             baselines=c_baselines,
             label="Completeness")

    # bbox_to_anchor=(1.1., 1, 0.3, .6),        (x, y, width, height)
    ax[1].legend(fontsize=14, bbox_to_anchor=(1.1, 1),
                 labelspacing=.2, handlelength=1)
    yticks = [0.22, 0.32]
    ax[0].set_yticks(np.array(yticks))
    new_labels = ["Ia (unspec.)", "II"]
    ax[0].set_yticklabels(new_labels,  fontsize=14, horizontalalignment='right')
    plt.gca().invert_yaxis()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_dir + "/LSST_Evaluation_Metrics.pdf",
                bbox_inches='tight')
