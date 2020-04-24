"""
Plotting code
"""
import numpy as np
import matplotlib.pyplot as plt
from estimate.constants import *


def plot_redshift_compare(data, labels, cname):
    """
    Plot redshift distributions of the subset of data for
    THEx all-features, THEx g-w2, LSST original, LSST filtered to match all-features, LSST to match g-w2
    """
    FIG_WIDTH = 6
    FIG_HEIGHT = 4
    DPI = 200
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                           dpi=DPI,
                           tight_layout=True,
                           sharex=True,
                           sharey=True)
    # Plotting parameters
    min_val = 0
    max_val = 1
    num_bins = NUM_BINS

    # Plotting visualization parameters
    small_lw = 0.8
    large_lw = 1.2
    LIGHT_BLUE = '#b3ccff'
    DARK_BLUE = '#003399'
    LIGHT_ORANGE = '#ffc2b3'
    DARK_ORANGE = '#ff3300'
    LIGHT_GREEN = '#b3ffcc'

    # LSST data
    ax.hist(data[2],
            color=LIGHT_GREEN,
            range=(min_val, max_val),
            bins=num_bins,
            histtype='step',
            linewidth=large_lw,
            density=True,
            label=labels[2])
    if len(data[0]) > 0:
        ax.hist(data[3],
                color=LIGHT_BLUE,
                range=(min_val, max_val),
                bins=num_bins,
                histtype='step',
                linewidth=large_lw,
                density=True,
                label=labels[3])
    ax.hist(data[4],
            color=LIGHT_ORANGE,
            range=(min_val, max_val),
            bins=num_bins,
            histtype='step',
            linewidth=large_lw,
            density=True,
            label=labels[4])

    # THEx data
    if len(data[0]) > 0:
        ax.hist(data[0],
                color=DARK_BLUE,
                range=(min_val, max_val),
                bins=num_bins,
                histtype='step',
                linewidth=small_lw,
                density=True,
                label=labels[0])
    ax.hist(data[1],
            color=DARK_ORANGE,
            range=(min_val, max_val),
            bins=num_bins,
            histtype='step',
            linewidth=small_lw,
            density=True,
            label=labels[1])

    ax.set_xlim(min_val, max_val)
    plt.legend(fontsize=10)
    plt.xlabel("Redshift", fontsize=10)
    cname = cname.replace("/", "_")
    print("saving figure to " + str(DATA_DIR + "../figures/" + cname + "_redshift_overlap.pdf"))
    plt.savefig(DATA_DIR + "../figures/" + cname + "_redshift_overlap.pdf")
    # plt.show()


def plot_fit(lsst_df, lsst_AF, lsst_GW2, lsst_AF_ranges, lsst_GW2_ranges, thex_Z_AF, thex_Z_gw2, class_name, lsst_feature_name):
    """
    :param lsst_df: Original LSST DataFrame
    :param lsst_AF: Numpy array of LSST filtered to all-features
    :param lsst_GW2: Numpy array of LSST filtered to g-W2-dataset
    :param lsst_GW2_ranges: [x,y] of range LSST was filtered on for g-W2
    :param thex_Z_AF: THEx redshift values for all-features
    :param thex_Z_gw2: THEx redshift values for g-W2
    :param class_name: THEx class name
    :param lsst_feature_name: LSST feature name
    """

    # Set all labels
    thex_Z_AF_label = "THEx all-features " + class_name
    thex_Z_gw2_label = "THEx g-W2 " + class_name

    lsst_data_orig = lsst_df["true_z"].values
    lsst_orig_label = "LSST " + class_name

    min_AF = round(lsst_AF_ranges[0], 2)
    max_AF = round(lsst_AF_ranges[1], 2)
    lsst_AF_label = "LSST all-features matching " + class_name + ", " + \
        lsst_feature_name + ": " + str(min_AF) + "<=" + str(max_AF)

    min_GW2 = round(lsst_GW2_ranges[0], 2)
    max_GW2 = round(lsst_GW2_ranges[1], 2)
    lsst_GW2_label = "LSST g-W2 matching  " + class_name + ", " + \
        lsst_feature_name + ": " + str(min_GW2) + "<=" + str(max_GW2)

    datas = [thex_Z_AF, thex_Z_gw2, lsst_data_orig, lsst_AF, lsst_GW2]
    labels = [thex_Z_AF_label, thex_Z_gw2_label,
              lsst_orig_label, lsst_AF_label, lsst_GW2_label]
    plot_redshift_compare(datas, labels, class_name)
