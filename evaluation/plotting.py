
import numpy as np
import matplotlib.pyplot as plt
from estimate.get_data import *
from thex_data.data_consts import *


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
