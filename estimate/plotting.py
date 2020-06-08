"""
Plotting code
"""
import numpy as np
import matplotlib.pyplot as plt
from estimate.constants import *


def plot_Z_ranges(title, data, LSST_label, file_title):
    """
    Plot 3 datasets, stored in data map: original THEx data, original LSST, and filtered LSST. All as redshift distributions for a certain class.
    """
    FIG_WIDTH = 6
    FIG_HEIGHT = 4
    DPI = 200
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI,
                           tight_layout=True, sharex=True, sharey=True)
    Z_bins = np.linspace(0, 1.1, 50)
    LIGHT_RED = "#ffb3b3"
    LIGHT_BLUE = "#99ccff"
    LIGHT_GREEN = "#99ffe6"
    DARK_BLUE = "#0000b3"
    ax.hist(data['THEx'], bins=Z_bins, density=True,
            label="THEx",  fill=False, edgecolor=DARK_BLUE)
    ax.hist(data['LSST_orig'], bins=Z_bins, density=True,
            label="Original LSST",  fill=False, edgecolor=LIGHT_GREEN)
    ax.hist(data['LSST_filt'], bins=Z_bins, density=True,
            label=LSST_label,  fill=False, edgecolor=LIGHT_BLUE)

    plt.title(title)
    xlabel = "Redshift"
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("../figures/analysis/" + file_title + ".png")
    plt.show()


def plot_redshift_compare(thex_data, lsst_orig, lsst_filt, lsst_filt_label, cname, dataset):
    """
    Plot redshift distributions of the subset of data versus original
    :param thex_data: THEx redshift distribution 
    :param lsst_orig: LSST redshfit distribution
    :param lsst_filt: LSST filtered to THEx
    :param lsst_filt_label: Label of range filtered to 
    :param cname: Class name
    :param dataset: Name of dataset
    THEx all-features, THEx g-w2, LSST original, LSST filtered to match all-features, LSST to match g-w2
    """

    orig_count = str(len(lsst_orig))
    filt_count = str(len(lsst_filt))
    thex_count = str(len(thex_data))
    print("Original number of samples in LSST: " + orig_count)
    print("Number of samples in LSST gw-2 filt: " + filt_count)

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
    LIGHT_ORANGE = "#ffddcc"
    DARK_ORANGE = '#ff5500'
    LIGHT_GREEN = '#b3ffcc'

    # Reset number of bins to less for smaller classes
    # use
    if len(thex_data) < 100:
        num_bins = 50

    ax.hist(lsst_orig,
            color=LIGHT_GREEN,
            range=(min_val, max_val),
            density=True,
            bins=num_bins,
            histtype='step',
            linewidth=large_lw,
            label="Target: Original (Count: " + orig_count + ")")
    ax.hist(lsst_filt,
            color=LIGHT_BLUE,
            range=(min_val, max_val),
            density=True,
            bins=num_bins,
            histtype='step',
            linewidth=large_lw,
            label=lsst_filt_label + " (Count: " + filt_count + ")")
    ax.hist(thex_data,
            color=DARK_BLUE,
            range=(min_val, max_val),
            density=True,
            bins=num_bins,
            histtype='step',
            linewidth=small_lw,
            label="THEx (Count: " + thex_count + ")")

    ax.set_xlim(min_val, max_val)
    plt.legend(fontsize=10)
    plt.xlabel("Redshift", fontsize=10)
    plt.title(cname)
    # plt.yscale('log')
    cname = cname.replace("/", "_")

    plt.savefig(DATA_DIR + "../figures/" + cname + "_" +
                str(dataset) + "_redshift_overlap.pdf")


def plot_cumsum(t, l, class_name):
    bins = np.linspace(0, 1.2, 100)
    t_length = len(t)
    l_length = len(l)
    l, binsl = np.histogram(l, bins=bins)
    t, binst = np.histogram(t, bins=bins)
    l = np.cumsum(l)
    t = np.cumsum(t)
    t = t / t_length
    l = l / l_length
    fig, ax = plt.subplots(tight_layout=True,
                           sharex=True,
                           sharey=True)
    ax.plot(bins[:-1], l, c='blue', label="LSST")
    ax.plot(bins[:-1], t, c='green', label="THEx")
    ax.legend()
    plt.ylabel("CDF")
    plt.xlabel("redshift")
    plt.title("CDFs for filtered " + class_name)
    plt.savefig(DATA_DIR + "../figures/" + class_name + "_" + "CDFs.pdf")
    plt.show()
