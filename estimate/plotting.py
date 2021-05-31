"""
Plotting code
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from estimate.constants import *

THEX_COLOR = "#ffa31a"
LSST_COLOR = "#80ccff"

DARK_BLUE = "#0000b3"
LIGHT_ORANGE = "#ffddcc"
DARK_ORANGE = '#ff5500'


def init_plot_settings():
    """
    Set defaults for all plots: font.
    """
    plt.rcParams["font.family"] = "Times New Roman"


def plot_Z_ranges_together(data, file_title):
    """
    Plots the Z dists of LSST vs THEx for each class.
    :param data: Dict from class name to map of {THEx : thex data, LSST_orig: LSST data}
    """
    FIG_WIDTH = 10
    FIG_HEIGHT = 5
    DPI = 200
    f, ax = plt.subplots(nrows=2,
                         ncols=3,
                         sharex=True, sharey=True,
                         figsize=(FIG_WIDTH, FIG_HEIGHT),
                         dpi=600)

    Z_bins = np.linspace(0, 1.1, 50)

    def plot_data(row_index, col_index, class_data, class_label):
        mpl.rcParams['hatch.linewidth'] = 1.0

        ax[row_index][col_index].hist(class_data['LSST_orig'],
                                      bins=Z_bins,
                                      density=True,
                                      label="LSST",
                                      fill=True,
                                      alpha=0.8,
                                      edgecolor=None,
                                      color=LSST_COLOR)
        # bins values in bin are (for left, right), left <= x < right
        vals, bins = np.histogram(class_data['THEx'], bins=Z_bins, density=True)
        a = np.array([0])
        bin_indices = np.linspace(min(Z_bins), max(Z_bins), len(Z_bins))
        bin_indices = bin_indices[1:]
        xnew = np.concatenate((a, bin_indices), axis=0)
        ynew = np.concatenate((a, vals), axis=0)
        ax[row_index][col_index].xaxis.set_tick_params(labelsize=16)
        ax[row_index][col_index].yaxis.set_tick_params(labelsize=16)
        ax[row_index][col_index].set_xlabel("Redshift", fontsize=18)
        ax[row_index][col_index].step(x=xnew,
                                      y=ynew,
                                      label="THEx",
                                      linewidth=2,
                                      color=THEX_COLOR)
        ax[row_index][col_index].set_title(class_label, fontsize=20, y=0.65, x=0.75)

        ax[row_index][col_index].tick_params(
            axis='x',           # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom=True,
            top=False,
            labelbottom=True)

    plot_data(0, 0, data["Unspecified Ia"], "Ia\n(unspec.)")
    plot_data(0, 1, data["Ia-91bg"], "Ia-91bg")
    plot_data(0, 2, data["Ibc"], "Ibc")

    plot_data(1, 0, data["Unspecified II"], "II\n(unspec.)")
    plot_data(1, 1, data["TDE"], "TDE")

    labelsize = 18
    ax[0][0].set_ylabel("Density", fontsize=labelsize)
    ax[1][0].set_ylabel("Density", fontsize=labelsize)
    # ax[1][2].axis('off')
    ax[0][2].set_visible(True)
    # ax[1][2].set_axis_off()
    f.delaxes(ax[1, 2])

    ax[0][2].xaxis.set_tick_params(labelsize=14)

    ax[0][0].legend(fontsize=16, loc="upper left", labelspacing=.2, handlelength=1)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(ROOT_DIR+"/output/" + file_title)


def plot_Z_ranges(title, data, file_title):
    """
    Plot 3 datasets, stored in data map: original THEx data, original LSST, and filtered LSST. All as redshift distributions for a certain class.
    """
    FIG_WIDTH = 6
    FIG_HEIGHT = 4
    DPI = 80
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI,
                           tight_layout=True, sharex=True, sharey=True)
    Z_bins = np.linspace(0, 1.1, 50)

    ax.hist(data['THEx'], bins=Z_bins, density=True,
            label="THEx",  fill=False, edgecolor=DARK_BLUE)
    ax.hist(data['LSST_orig'], bins=Z_bins, density=True,
            label="LSST",  fill=False, edgecolor=LSST_COLOR)

    # ax.hist(data['LSST_filt'], bins=Z_bins, density=True,
    #         label=LSST_label,  fill=False, edgecolor=THEX_COLOR)

    plt.title(title, fontsize=16)
    xlabel = "Redshift"
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=16)
    plt.savefig("../figures/analysis/" + file_title)


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

    # Reset number of bins to less for smaller classes
    # use
    if len(thex_data) < 100:
        num_bins = 50

    ax.hist(lsst_orig,
            color=LSST_COLOR,
            range=(min_val, max_val),
            density=True,
            bins=num_bins,
            histtype='step',
            linewidth=large_lw,
            label="Target: Original (Count: " + orig_count + ")")
    ax.hist(lsst_filt,
            color=THEX_COLOR,
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
