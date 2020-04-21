#!/usr/bin/env python
# coding: utf-8

from functools import partial
import multiprocessing

import pickle
import pandas as pd
import numpy as np


from estimate.constants import *


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


def filter_lsst_data(feature_name, min_feature, max_feature, data):
    """
    Get redshift values for data filtered on this range: label class_name
    whose  feature values are in the range [min_feature,max_feature ] 
    :param data: class data
    """

    feature_vals = data[feature_name]
    indices = []
    for index, f in enumerate(data[feature_name]):
        if ~np.isnan(f):
            if f >= min_feature and f <= max_feature:
                indices.append(f)

    class_data_Z = data['true_z']

    class_data_filt = np.take(class_data_Z, indices)

    class_data_filt = class_data_filt[~np.isnan(class_data_filt)]
    return class_data_filt


def get_hist(data):
    """
    Get histograms for the two datas
    """
    hist, bin_edges = np.histogram(data, range=(0, 1),
                                   bins=NUM_BINS, density=True)
    return hist


def get_fit(lsst_data, thex_hist, class_name, lsst_feature_name, test_range):
    """
    Estimate RMSE for min and max range for feature and class between LSST and THEx redshift distribution
    """

    print("Testing range " + str(test_range))
    min_range = test_range[0]
    max_range = test_range[1]
    lsst_data_filt = filter_lsst_data(feature_name=lsst_feature_name,
                                      min_feature=min_range,
                                      max_feature=max_range,
                                      data=lsst_data)

    if len(lsst_data_filt) == 0:
        return 100
    lsst_hist = get_hist(lsst_data_filt)
    t = np.array(thex_hist)
    l = np.array(lsst_hist)

    rmse = np.sqrt(np.mean((t - l)**2))
    return rmse


def get_best_range(lsst_class_data, thex_data_df, class_name, feature_name, lsst_feature_name, min_vals, max_vals):
    """
    Find min and max range for feature that gets LSST and THEx redshift distributions for this class as close together, as measured by RMSE.
    """

    ranges = []
    for min_range in min_vals:
        for max_range in max_vals:
            if min_range < max_range:
                ranges.append([min_range, max_range])

    thex_data_orig = get_thex_class_redshifts(class_name, thex_data_df)
    thex_hist = get_hist(thex_data_orig)

    print("Running " + str(CPU_COUNT) + " processes.")
    pool = multiprocessing.Pool(CPU_COUNT)

    # Pass in parameters that don't change for parallel processes
    func = partial(get_fit,
                   lsst_class_data,
                   thex_hist,
                   class_name,
                   lsst_feature_name)
    # Multithread over ranges
    rmses = []
    rmses = pool.map(func, ranges)
    pool.close()
    pool.join()
    print("Done processing...")

    min_rmse_index = rmses.index(min(rmses))
    best_rmse = rmses[min_rmse_index]
    best_range = ranges[min_rmse_index]

    best_min = best_range[0]
    best_max = best_range[1]
    print("Best RMSE: " + str(best_rmse) + " with range " +
          str(best_min) + " - " + str(best_max))
    return best_min, best_max

from scipy import stats
import math


def get_KS_fit(lsst_data, thex_data, class_name, lsst_feature_name, test_range):
    """
    Estimate RMSE for min and max range for feature and class between LSST and THEx redshift distribution
    """

    print("Testing range " + str(test_range))
    min_range = test_range[0]
    max_range = test_range[1]

    lsst_data_filt = filter_lsst_data(feature_name=lsst_feature_name,
                                      min_feature=min_range,
                                      max_feature=max_range,
                                      data=lsst_data)

    if len(lsst_data_filt) == 0:
        return 100, 0, False

    l = lsst_data_filt
    t = thex_data

    KS_statistic, p_value = stats.ks_2samp(t, l)

    t_size = len(t)
    l_size = len(l)

    D_critical = 1.36 * math.sqrt((t_size + l_size) / (t_size * l_size))

    acceptable = False
    if KS_statistic < D_critical:
        print(str(KS_statistic) + " for D critical " + str(D_critical))
        acceptable = True

    print("\nKS Stat: " + str(KS_statistic) + "; P-value: " +
          str(p_value) + "; Accepted " + str(acceptable))
    return [KS_statistic, p_value, acceptable]


def get_best_KS_range(lsst_class_data, thex_data_df, class_name, feature_name, lsst_feature_name, min_vals, max_vals):
    """
    Find min and max range for feature that gets LSST and THEx redshift distributions for this class as close together, as measured by RMSE.
    """

    ranges = []
    for min_range in min_vals:
        for max_range in max_vals:
            if min_range < max_range:
                ranges.append([min_range, max_range])

    thex_data_orig = get_thex_class_redshifts(class_name, thex_data_df)

    # print("Running " + str(CPU_COUNT) + " processes.")
    # pool = multiprocessing.Pool(CPU_COUNT)
    # # Pass in parameters that don't change for parallel processes
    # func = partial(get_KS_fit, lsst_class_data,
    #                thex_data_orig,
    #                class_name,
    #                lsst_feature_name)
    # # Multithread over ranges
    # stats = []
    # stats = pool.map(func, ranges)
    # pool.close()
    # pool.join()
    # print("Done processing...")

    stats = []
    p_values = []
    accepted = []  # True if D < D_critical
    for r in ranges:
        print(r)
        stat, p, a = get_KS_fit(lsst_class_data,
                                thex_data_orig,
                                class_name,
                                lsst_feature_name,
                                r)
        stats.append(stat)
        p_values.append(p)
        accepted.append(a)

    # Select KS with lowest value
    min_stat_index = stats.index(min(stats))
    best_stat = stats[min_stat_index]
    best_range = ranges[min_stat_index]
    p_value = p_values[min_stat_index]
    accepted_value = accepted[min_stat_index]

    best_min = best_range[0]
    best_max = best_range[1]
    print("\n\nBest range: " + str(best_min) + " - " + str(best_max))
    print("KS: " + str(best_stat) + "\np= " +
          str(p_value) + "\nAccepted: " + str(accepted_value))
    return best_min, best_max


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

    # Call like python estimate_dists.py Ia Ia r

    # Pull down LSST-like data
    with open(DATA_DIR + 'lsst-sims.pk', 'rb') as f:
        lc_prop = pickle.load(f)
    # Pull down our data
    df_all_features = get_data(name='all_features')
    df_g_W2 = get_data(name='g_W2')

    # Only features to choose from are g, r, i, z, y
    thex_class_name = argv[1]  # 'Ib/c'
    lsst_class_name = argv[2]  # 'Ibc'
    feature = argv[3]  # 'r_mag'

    feature_name = feature + '_mag'
    lsst_feature_name = feature + '_first_mag'
    num_samples = 20
    min_vals = np.linspace(8, 18, num_samples)
    max_vals = np.linspace(20, 30, num_samples)

    lsst_class_data = lc_prop[lsst_class_name]

    # All Features best params
    print("\nEstimating for all-features dataset")
    best_min_AF, best_max_AF = get_best_KS_range(lsst_class_data,
                                                 df_all_features,
                                                 thex_class_name,
                                                 feature_name,
                                                 lsst_feature_name,
                                                 min_vals,
                                                 max_vals)
    # g-W2 dataset best params
    print("\nEstimating for g-W2-dataset")
    best_min_GW2, best_max_GW2 = get_best_KS_range(lsst_class_data,
                                                   df_g_W2,
                                                   thex_class_name,
                                                   feature_name,
                                                   lsst_feature_name,
                                                   min_vals,
                                                   max_vals)

    lsst_data_AF = filter_lsst_data(feature_name=lsst_feature_name,
                                    min_feature=best_min_AF,
                                    max_feature=best_max_AF,
                                    data=lsst_class_data)

    lsst_data_gw2 = filter_lsst_data(feature_name=lsst_feature_name,
                                     min_feature=best_min_GW2,
                                     max_feature=best_max_GW2,
                                     data=lsst_class_data)

    lsst_AF_ranges = [best_min_AF, best_max_AF]
    lsst_GW2_ranges = [best_min_GW2, best_max_GW2]

    plot_fit(lsst_class_data,
             lsst_data_AF,
             lsst_data_gw2,
             lsst_AF_ranges,
             lsst_GW2_ranges,
             df_all_features,
             df_g_W2,
             thex_class_name,
             lsst_feature_name)

    # chisq, p = chisquare(f_obs=thex_hist, f_exp=lsst_hist)
    # print("Chi squared test statistic: " + str(chisq))
    # print("p value: " + str(p))


if __name__ == "__main__":
    main(sys.argv)
