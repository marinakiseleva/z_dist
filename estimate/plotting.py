"""
Plotting code
"""
import numpy as np
import matplotlib.pyplot as plt
from estimate.constants import *


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
            bins=num_bins,
            histtype='step',
            linewidth=large_lw,
            # density=True,
            label="Target: Original")

    ax.hist(lsst_filt,
            color=LIGHT_BLUE,
            range=(min_val, max_val),
            bins=num_bins,
            histtype='step',
            linewidth=large_lw,
            # density=True,
            label=lsst_filt_label)
    ax.hist(thex_data,
            color=DARK_BLUE,
            range=(min_val, max_val),
            bins=num_bins,
            histtype='step',
            linewidth=small_lw,
            # density=True,
            label="THEx")

    ax.set_xlim(min_val, max_val)
    plt.legend(fontsize=10)
    plt.xlabel("Redshift", fontsize=10)
    plt.title(cname)
    plt.yscale('log')
    cname = cname.replace("/", "_")

    plt.savefig(DATA_DIR + "../figures/" + cname + "_" +
                str(dataset) + "_redshift_overlap.pdf")
