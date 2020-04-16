#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chisquare

TARGET_LABEL = 'transient_type'
DATA_DIR = 'data/'


def get_data(name):
    """
    Pull down project data, one of two types:
    all-features-dataset: 'all_features'
    g_W2-dataset: 'g_W2'
    """
    X = pd.read_csv(DATA_DIR + name + '_X.csv')
    X.drop(['Unnamed: 0'], axis=1, inplace=True)
    y = pd.read_csv(DATA_DIR + name + '_y.csv')
    y.drop(['Unnamed: 0'], axis=1, inplace=True)
    df = pd.concat([X, y], axis=1)
    return df


def convert_str_to_list(input_string):
    """
    Convert string to list
    """
    l = input_string.split(",")
    return [item.strip(' ') for item in l]


def get_thex_class_redshifts(class_name, data):
    keep_indices = []
    for index, row in data.iterrows():
        labels = convert_str_to_list(row[TARGET_LABEL])
        if class_name in labels:
            keep_indices.append(index)

    return data.loc[keep_indices, :]['redshift'].values


def get_thex_class_count(class_name, data):

    keep_indices = []
    for index, row in data.iterrows():
        labels = convert_str_to_list(row[TARGET_LABEL])
        if class_name in labels:
            keep_indices.append(index)

    return data.loc[keep_indices, :].reset_index(drop=True)


def filter_class_feature_range(class_name, feature_name, min_feature, max_feature, data):
    """
    Get redshift values for data filtered on this range:
    Filter data to those with class value and feature in range. 
    :param class_name: class name to filter on
    :param feature_name: Name of feature to filter on
    :param min_feature: Minimum feature value to keep
    :param max_feature: Maximum feature value to keep
    :param data: Pandas DataFrame from model 
    """

    keep_indices = []
    for index, row in data.iterrows():
        labels = convert_str_to_list(row[TARGET_LABEL])
        in_range = row[feature_name] >= min_feature and row[feature_name] <= max_feature
        if class_name in labels and in_range:
            keep_indices.append(index)

    filt_df = data.loc[keep_indices, :].reset_index(drop=True)

    return filt_df['redshift'].values


def filter_lsst_data(class_name, feature_name, min_feature, max_feature, data):
    """
    Get redshift values for data filtered on this range: label class_name
    whose  feature values are in the range [min_feature,max_feature ] 
    :param data: loaded pickle of lsst-sims.pk
    """
    class_data = data[class_name]

    feature_vals = class_data[feature_name]
    indices = np.where((feature_vals >= min_feature) & (feature_vals <= max_feature))[0]

    class_data_Z = class_data['true_z']

    class_data_filt = np.take(class_data_Z, indices)

    class_data_filt = class_data_filt[~np.isnan(class_data_filt)]

    return class_data_filt


def get_best_range(lsst_data, thex_data_df, class_name, feature_name, lsst_feature_name, min_vals, max_vals):
    best_rmse = 100
    best_range = 0
    for min_range in min_vals:
        for max_range in max_vals:
            if min_range < max_range:
                rmse = get_fit(lsst_data,
                               thex_data_df,
                               class_name,
                               feature_name,
                               lsst_feature_name,
                               min_range,
                               max_range)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_range = [min_range, max_range]

    best_min = best_range[0]
    best_max = best_range[1]
    print("Best RMSE: " + str(best_rmse) + " with range " +
          str(best_min) + " - " + str(best_max))
    return best_min, best_max


def get_hists(thex_data, lsst_data_filt):
    """
    Get histograms for the two datas
    """
    min_val = min(thex_data.min(), lsst_data_filt.min())
    max_val = 1  # max(thex_data.max(), lsst_data.max())
    num_bins = 200
    thex_hist, bin_edges = np.histogram(thex_data,
                                        range=(min_val, max_val),
                                        bins=num_bins,
                                        density=True)

    lsst_hist, bin_edges = np.histogram(lsst_data_filt,
                                        range=(min_val, max_val),
                                        bins=num_bins,
                                        density=True)
    return thex_hist, lsst_hist


def get_fit(lsst_data, thex_data_df, class_name, feature_name, lsst_feature_name, min_range, max_range):
    lsst_data_filt = filter_lsst_data(class_name=class_name,
                                      feature_name=lsst_feature_name,
                                      min_feature=min_range,
                                      max_feature=max_range,
                                      data=lsst_data)

    thex_data_orig = get_thex_class_redshifts(class_name, thex_data_df)

    if len(thex_data_orig) == 0 or len(lsst_data_filt) == 0:
        return 100
    thex_hist, lsst_hist = get_hists(thex_data_orig, lsst_data_filt)
    t = np.array(thex_hist)
    l = np.array(lsst_hist)

    rmse = np.sqrt(np.mean((t - l)**2))
    return rmse


"""
Plotting code
"""


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
    num_bins = 100

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
    plt.savefig("figures/" + cname + "_redshift_overlap.png")
    plt.show()


def plot_fit(lsst_orig, lsst_AF, lsst_GW2, lsst_AF_ranges, lsst_GW2_ranges, thex_data_AF, thex_data_gW2, class_name, lsst_feature_name):

    thex_Z_AF_label = "THEx all-features " + class_name
    thex_Z_gw2_label = "THEx g-W2 " + class_name
    thex_Z_AF = get_thex_class_redshifts(class_name, thex_data_AF)
    thex_Z_gw2 = get_thex_class_redshifts(class_name, thex_data_gW2)

    lsst_data_orig = lsst_orig[class_name]["true_z"]
    lsst_data_orig = lsst_data_orig[~np.isnan(lsst_data_orig)]
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


# keys in lsst-sims.pk are:
# obj_id:                         light curve id
# true_z, photo_z:                transient redshift and host photo-z

# These columns are calculated for each band (* = u, g, r, i, z, y)

# *_first_mjd:                    epoch of initial detection ('first epoch')
# *_first_snr:                    first-epoch SNR
# *_min_snr, *_max_snr:           minimal and maximal SNR of the light curve
# *_first_mag, *_first_mag_err:   first-epoch magnitude and error
# *_min_mag, *_min_mag_err:       faintest magnitude and error
# *_max_mag, *_max_mag_err:       peak or brighest magnitude and error
# *_first_flux, *_first_flux_err: first-epoch physical flux and error
# *_min_flux, *_min_flux_err:     minimal flux (matching faintest magnitude)
# *_max_flux, *_max_flux_err:     maximal flux (matching peak magnitude)

import sys


def main(argv):
    # Pull down data

    with open(DATA_DIR + 'lsst-sims.pk', 'rb') as f:
        lc_prop = pickle.load(f)

    df_all_features = get_data(name='all_features')

    df_g_W2 = get_data(name='g_W2')

    # lc_prop.keys()
    # Only features to choose from are g, r, i, z, y

    class_name = argv[1]  # 'Ia'
    feature = argv[2]  # 'r_mag'

    feature_name = feature + '_mag'
    lsst_feature_name = feature + '_first_mag'
    min_vals = np.linspace(10, 12, 20)
    max_vals = np.linspace(15, 20, 20)

    # All Features best params
    print("\nEstimating for all-features dataset")
    best_min_AF, best_max_AF = get_best_range(lc_prop,
                                              df_all_features,
                                              class_name,
                                              feature_name,
                                              lsst_feature_name,
                                              min_vals,
                                              max_vals)
    # g-W2 dataset best params
    print("\nEstimating for g-W2-dataset")
    best_min_GW2, best_max_GW2 = get_best_range(lc_prop,
                                                df_g_W2,
                                                class_name,
                                                feature_name,
                                                lsst_feature_name,
                                                min_vals,
                                                max_vals)

    lsst_data_AF = filter_lsst_data(class_name=class_name,
                                    feature_name=lsst_feature_name,
                                    min_feature=best_min_AF,
                                    max_feature=best_max_AF,
                                    data=lc_prop)

    lsst_data_gw2 = filter_lsst_data(class_name=class_name,
                                     feature_name=lsst_feature_name,
                                     min_feature=best_min_GW2,
                                     max_feature=best_max_GW2,
                                     data=lc_prop)

    lsst_AF_ranges = [best_min_AF, best_max_AF]
    lsst_GW2_ranges = [best_min_GW2, best_max_GW2]

    plot_fit(lc_prop,
             lsst_data_AF,
             lsst_data_gw2,
             lsst_AF_ranges,
             lsst_GW2_ranges,
             df_all_features,
             df_g_W2,
             class_name,
             lsst_feature_name)

    # chisq, p = chisquare(f_obs=thex_hist, f_exp=lsst_hist)
    # print("Chi squared test statistic: " + str(chisq))
    # print("p value: " + str(p))


if __name__ == "__main__":
    main(sys.argv)
