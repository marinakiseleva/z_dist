#!/usr/bin/env python
# coding: utf-8

from functools import partial
import multiprocessing

import pickle
import pandas as pd
import numpy as np
from scipy import stats
import math

from estimate.constants import *
from estimate.plotting import *


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


def get_thex_class_data(class_name, data):
    """
    Filter DataFrame to have only data with label class_name
    """
    keep_indices = []
    for index, row in data.iterrows():
        labels = convert_str_to_list(row[TARGET_LABEL])
        if class_name in labels:
            keep_indices.append(index)

    return data.loc[keep_indices, :]


def get_thex_class_redshifts(class_name, data):
    return get_thex_class_data(class_name, data)['redshift'].values


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


def get_lsst_class_data(class_name, feature_name, data):
    """
    Get LSST data with this class name, and valid values for feature name
    """

    lsst_class_data = data[class_name]

    feature_data = lsst_class_data[feature_name]

    indices = []
    for index, f in enumerate(feature_data):
        if ~np.isnan(f):
            indices.append(index)

    valid_mags = np.take(lsst_class_data[feature_name], indices)
    valid_Z = np.take(lsst_class_data['true_z'], indices)

    df = pd.DataFrame(valid_mags, columns=[feature_name])
    df['true_z'] = valid_Z.tolist()

    return df


def get_LSST_filt_redshifts(min_feature, max_feature, data):
    """
    Get redshift values as Numpy array, for feature values in the range [min_feature, max_feature]. First column is features.
    :param feature_name: Name of feature to filter on
    :param min_feature: Minimum acceptable value of feature
    :param max_feature: Maximum acceptable value of feature
    :param data: LSST data as DataFrame with feature and redshift column
    """
    f_df = data[(data.iloc[:, 0] >= min_feature) & (data.iloc[:, 0] <= max_feature)]

    return f_df['true_z'].values


def get_KS_fit(lsst_df, thex_data, test_range):
    """
    Estimate RMSE for min and max range for feature and class between LSST and THEx redshift distribution
    """

    # print("Testing range " + str(test_range))
    min_range = test_range[0]
    max_range = test_range[1]

    lsst_data_filt = get_LSST_filt_redshifts(min_feature=min_range,
                                             max_feature=max_range,
                                             data=lsst_df)

    if len(lsst_data_filt) == 0:
        return 100, 0, False

    l = lsst_data_filt
    t = thex_data

    KS_statistic, p_value = stats.ks_2samp(t, l)

    t_size = len(t)
    l_size = len(l)

    D_critical = 1.95 * math.sqrt((t_size + l_size) / (t_size * l_size))

    acceptable = False
    if KS_statistic < D_critical:
        print("\nFor range " + str(min_range) + " - " + str(max_range) +
              "\nAccepted " + str(KS_statistic) + " for D critical " + str(D_critical) + "; P-value: " + str(p_value))
        acceptable = True

    return [KS_statistic, p_value, acceptable]


def get_best_KS_range(lsst_df, thex_redshifts, min_vals, max_vals):
    """
    Find min and max range for feature that gets LSST and THEx redshift distributions for this class as close together, as measured by RMSE.
    :param lsst_data: LSST DataFrame of redshift and feature values for class
    :param thex_redshifts: List of THEx redshifts
    """

    ranges = []
    for min_range in min_vals:
        for max_range in max_vals:
            if min_range < max_range:
                ranges.append([min_range, max_range])

    print("Running " + str(CPU_COUNT) + " processes.")
    pool = multiprocessing.Pool(CPU_COUNT)
    # Pass in parameters that don't change for parallel processes
    func = partial(get_KS_fit,
                   lsst_df,
                   thex_redshifts)
    # Multithread over ranges
    overall_stats = []
    overall_stats = pool.map(func, ranges)
    pool.close()
    pool.join()
    print("Done processing...")
    stats = []
    p_values = []
    accepted = []  # True if D < D_critical
    for s in overall_stats:
        stats.append(s[0])
        p_values.append(s[1])
        accepted.append(s[2])

    # stats = []
    # p_values = []
    # accepted = []  # True if D < D_critical
    # for r in ranges:
    #     print(r)
    #     stat, p, a = get_KS_fit(lsst_df, thex_redshifts, r)
    #     stats.append(stat)
    #     p_values.append(p)
    #     accepted.append(a)

    # Select range with lowest KS
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

    # Alternatively: find largest range with True acceptance (pass test)
    max_range = 0
    best_r_index = None  # Index corresponding to largest true range
    for index, a in enumerate(accepted):
        if a == True:
            min_val, max_val = ranges[index]
            cur_range = max_val - min_val
            if cur_range > max_range:
                max_range = cur_range
                best_r_index = index
    if best_r_index is not None:
        best_range = ranges[best_r_index]
        best_min = best_range[0]
        best_max = best_range[1]
        print("\n\nBest range according to max True range: " +
              str(best_min) + " - " + str(best_max))
        print("KS: " + str(stats[best_r_index]) + "\np= " +
              str(p_values[best_r_index]) + "\nAccepted: " + str(accepted[best_r_index]))

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


def get_thex_z_data(thex_class_name):
    """
    Pull down our data, filter on class name
    """
    df_all_features = get_data(name='all_features')
    df_g_W2 = get_data(name='g_W2')

    thex_Z_AF = get_thex_class_redshifts(thex_class_name, df_all_features)
    thex_Z_gw2 = get_thex_class_redshifts(thex_class_name, df_g_W2)

    return thex_Z_AF, thex_Z_gw2


def main(argv):
    """
    Call like python estimate_dists.py Ia Ia r
    Only features to choose from are g, r, i, z, y
    """
    # Initialize passed-in args
    thex_class_name = argv[1]
    lsst_class_name = argv[2]
    feature = argv[3]
    feature_name = feature + '_mag'
    lsst_feature_name = feature + '_first_mag'

    # Pull down LSST-like data
    with open(DATA_DIR + 'lsst-sims.pk', 'rb') as f:
        lc_prop = pickle.load(f)
    lsst_df = get_lsst_class_data(class_name=lsst_class_name,
                                  feature_name=lsst_feature_name,
                                  data=lc_prop)

    # Pull down our data
    thex_Z_AF, thex_Z_gw2 = get_thex_z_data(thex_class_name)

    # Set ranges of values to search over
    num_samples = 100
    # min_vals = [15.99, 16.03]
    # max_vals = [16.3, 16.24]
    # min_vals = [13]
    # max_vals = [18.3, 18.55, 21.15]

    min_vals = np.linspace(13, 17, num_samples)
    max_vals = np.linspace(16, 22, num_samples)

    delim = "-" * 100
    lsst_Z_orig = lsst_df["true_z"].values

    # All Features best params
    if len(thex_Z_AF) > 25:
        print(delim + "\nEstimating for all-features dataset")
        best_min_AF, best_max_AF = get_best_KS_range(lsst_df=lsst_df,
                                                     thex_redshifts=thex_Z_AF,
                                                     min_vals=min_vals,
                                                     max_vals=max_vals)
        lsst_data_AF = get_LSST_filt_redshifts(min_feature=best_min_AF,
                                               max_feature=best_max_AF,
                                               data=lsst_df)
        lsst_AF_ranges = [best_min_AF, best_max_AF]
        plot_redshift_compare(thex_data=thex_Z_AF,
                              lsst_orig=lsst_Z_orig,
                              lsst_filt=lsst_data_AF,
                              lsst_range=lsst_AF_ranges,
                              cname=thex_class_name,
                              dataset='allfeatures')

    # g-W2 dataset best params
    print(delim + "\nEstimating for g-W2-dataset")
    best_min_GW2, best_max_GW2 = get_best_KS_range(lsst_df=lsst_df,
                                                   thex_redshifts=thex_Z_gw2,
                                                   min_vals=min_vals,
                                                   max_vals=max_vals)

    lsst_data_gw2 = get_LSST_filt_redshifts(min_feature=best_min_GW2,
                                            max_feature=best_max_GW2,
                                            data=lsst_df)

    lsst_GW2_ranges = [best_min_GW2, best_max_GW2]

    plot_redshift_compare(thex_data=thex_Z_gw2,
                          lsst_orig=lsst_Z_orig,
                          lsst_filt=lsst_data_gw2,
                          lsst_range=lsst_GW2_ranges,
                          cname=thex_class_name,
                          dataset='gW2')


if __name__ == "__main__":
    main(sys.argv)
