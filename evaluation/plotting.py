
import numpy as np
import matplotlib.pyplot as plt


def init_plot_settings():
    """
    Set defaults for all plots: font.
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


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
