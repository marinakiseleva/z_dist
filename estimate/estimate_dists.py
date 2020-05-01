# !/usr / bin / env python
# coding: utf-8
import sys
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


def get_thex_z_data(class_name):
    """
    Pull down our data, filter on class name
    """
    df_AF = get_data(name='all_features')
    df_g_W2 = get_data(name='g_W2')

    thex_AF_Z = get_thex_class_data(class_name, df_AF)['redshift'].values
    thex_gw2_Z = get_thex_class_data(class_name, df_g_W2)['redshift'].values

    return thex_AF_Z, thex_gw2_Z


def get_lsst_class_data(class_name, feature_name, data):
    """
    Filter LSST data to only those samples with this class name, and valid values for feature name. Return as Pandas DataFrame with first column as feature values and second column as z
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


def get_LSST_filt_redshifts(min_feature, max_feature, data, r2=None):
    """
    Get redshift values as Numpy array, for feature values in the range [min_feature, max_feature]. First column is features.
    :param feature_name: Name of feature to filter on
    :param min_feature: Minimum acceptable value of feature
    :param max_feature: Maximum acceptable value of feature
    :param data: LSST data as DataFrame with feature and redshift column
    :param r2: Second range to include, if not None
    """
    if r2 is None:
        f_df = data[(data.iloc[:, 0] >= min_feature) & (data.iloc[:, 0] <= max_feature)]
    else:
        r2_min = r2[0]
        r2_max = r2[1]
        f_df = data[((data.iloc[:, 0] >= min_feature) & (data.iloc[:, 0] <= max_feature)) | (
            (data.iloc[:, 0] >= r2_min) & (data.iloc[:, 0] <= r2_max))]

    return f_df['true_z'].values


def get_stats(l, t):
    """
    Get KS test statistic, p-value, and where its accepted based on critical value between 2 datasets
    """
    if len(l) == 0:
        return 100, 0, False

    KS_statistic, p_value = stats.ks_2samp(t, l)

    t_size = len(t)
    l_size = len(l)

    alpha = 1.95
    D_critical = alpha * math.sqrt((t_size + l_size) / (t_size * l_size))

    acceptable = False
    if KS_statistic < D_critical:
        # print("\nAccepted " + str(KS_statistic) + " for D critical " +
        #       str(D_critical) + "; P-value: " + str(p_value))
        acceptable = True

    return [KS_statistic, p_value, acceptable]


def get_count(lsst_data, min_val, max_val, r2_range):
    """
    Get number of samples in LSST data in this range. 
    :param r2_range: None or [x,y]
    """
    vals = get_LSST_filt_redshifts(min_feature=min_val,
                                   max_feature=max_val,
                                   data=lsst_data,
                                   r2=r2_range)
    return len(vals)


def get_best_range_index(ranges, stats, p_values, accepted, lsst_data, r2s=None):
    """
    Get index of best range (with most samples), by maximizing p-value or meeting critical acceptance from KS test. 
    """
    max_count = 0
    best_p = 0
    best_r_index = None    # Index corresponding to largest true range
    p_threshold = .4
    for index, p in enumerate(p_values):
        min_val, max_val = ranges[index]
        cur_range = max_val - min_val
        r2_range = None
        if r2s is not None:
            # Add in range of range2
            min_val2, max_val2 = r2s[index]
            # Values are None when using no double range, (we run no double range to
            # compare to double ranges)
            if min_val2 is not None and max_val2 is not None:
                r2_range = [min_val2, max_val2]

        sample_count = get_count(lsst_data, min_val, max_val, r2_range)
        if (p >= p_threshold or accepted[index]) and sample_count >= max_count:
            max_count = sample_count
            best_p = p
            best_r_index = index

    return best_r_index


def get_best_range(ranges, stats, p_values, accepted, lsst_data, r2s=None):
    """
    Get best range: maximizes p-value and range.  
    Changed from before where we only considered those with 'true' accepted values (since they pass the KS test)
    :param stats: KS test statistics (per range)
    :param p_values: P values (per range)
    :param accepted: Booleans of acceptance (< D_critical) (per range)
    """
    best_r_index = get_best_range_index(
        ranges, stats, p_values, accepted, lsst_data, r2s)
    if best_r_index is not None:
        best_min = ranges[best_r_index][0]
        best_max = ranges[best_r_index][1]

        print("\n\nBest range " + str(best_min) + " - " + str(best_max))
        print("KS: " + str(stats[best_r_index]) + "\np= " +
              str(p_values[best_r_index]) + "\nAccepted: " + str(accepted[best_r_index]))
        if r2s is not None:
            print("With range 2: " + str(r2s[best_r_index]))
            return best_min, best_max, r2s[best_r_index]
    else:
        raise ValueError("No range accepted.")
    return best_min, best_max


def get_KS_double_fit(lsst_df, thex_data, ranges, index):
    """
    Get fit when using two ranges for the feature
    """
    print("Current range " + str(index + 1) + "/" + str(len(ranges)))
    stats = []
    r1_min = ranges[index][0]
    r1_max = ranges[index][1]

    for range2 in ranges:
        r2_min = range2[0]
        r2_max = range2[1]
        if r1_max < r2_min:
            lsst_1 = get_LSST_filt_redshifts(min_feature=r1_min,
                                             max_feature=r1_max,
                                             data=lsst_df)
            lsst_2 = get_LSST_filt_redshifts(min_feature=r2_min,
                                             max_feature=r2_max,
                                             data=lsst_df)
            lsst_data_filt = np.concatenate((lsst_1, lsst_2))
            k, p, a = get_stats(l=lsst_data_filt, t=thex_data)
            stats.append([k, p, a, range2])

    # Get stats for single range, to compare
    lsst = get_LSST_filt_redshifts(min_feature=r1_min,
                                   max_feature=r1_max,
                                   data=lsst_df)
    k, p, a = get_stats(l=lsst, t=thex_data)
    stats.append([k, p, a, [None, None]])

    range1s = [[r1_min, r1_max] for x in range(len(ranges))]
    r_stats = np.array(stats)
    best_index = get_best_range_index(ranges=range1s,
                                      stats=r_stats[:, 0],
                                      p_values=r_stats[:, 1],
                                      accepted=r_stats[:, 2],
                                      r2s=r_stats[:, 3],
                                      lsst_data=lsst_df)
    if best_index is None:
        return [100, 0, False, [None, None]]
    else:
        best_stats = list(r_stats[best_index, :])
        return best_stats


def get_KS_fit(lsst_df, thex_data, test_range):
    """
    Estimate RMSE for min and max range for feature and class between LSST and THEx redshift distribution
    """

    min_range = test_range[0]
    max_range = test_range[1]

    lsst_data_filt = get_LSST_filt_redshifts(min_feature=min_range,
                                             max_feature=max_range,
                                             data=lsst_df)

    stats = get_stats(l=lsst_data_filt, t=thex_data)

    return stats


def get_ranges(min_vals, max_vals):
    """
    Get ranges from min and max vals
    """
    ranges = []
    for min_range in min_vals:
        for max_range in max_vals:
            if min_range < max_range:
                ranges.append([min_range, max_range])
    print("Number of ranges " + str(len(ranges)))
    return ranges


def get_best_KS_range(lsst_df, thex_redshifts, min_vals, max_vals):
    """
    Find min and max range for feature that gets LSST and THEx redshift distributions for this class as close together, as measured by RMSE.
    :param lsst_data: LSST DataFrame of redshift and feature values for class
    :param thex_redshifts: List of THEx redshifts
    """
    ranges = get_ranges(min_vals, max_vals)

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

    best_min, best_max = get_best_range(ranges, stats, p_values, accepted, lsst_df)

    return best_min, best_max, None


def get_best_KS_double_range(lsst_df, thex_redshifts, min_vals, max_vals):
    """
    Same as above but can use 2 ranges.
    :param lsst_data: LSST DataFrame of redshift and feature values for class
    :param thex_redshifts: List of THEx redshifts
    """
    ranges = get_ranges(min_vals, max_vals)

    print("Running " + str(CPU_COUNT) + " processes.")
    pool = multiprocessing.Pool(CPU_COUNT)
    # Pass in parameters that don't change for parallel processes
    func = partial(get_KS_double_fit,
                   lsst_df,
                   thex_redshifts,
                   ranges)
    indices = [i for i in range(len(ranges))]
    # Multithread over ranges
    overall_stats = []
    overall_stats = pool.map(func, indices)
    pool.close()
    pool.join()
    print("Done processing...")
    stats = []
    p_values = []
    accepted = []  # True if D < D_critical
    range2s = []
    for s in overall_stats:
        stats.append(s[0])
        p_values.append(s[1])
        accepted.append(s[2])
        range2s.append(s[3])

    # stats = []
    # p_values = []
    # accepted = []  # True if D < D_critical
    # range2s = []
    # for r in ranges:
    #     print(r)
    #     stat, p, a, r2 = get_KS_double_fit(lsst_df, thex_redshifts, ranges, r)
    #     stats.append(stat)
    #     p_values.append(p)
    #     accepted.append(a)
    #     range2s.append(r2)

    r1_min, r1_max, r2 = get_best_range(
        ranges, stats, p_values, accepted, lsst_df, range2s)
    r2_min = r2[0]
    r2_max = r2[1]
    print("Best range " + str(r1_min) + " <= r1 <= " +
          str(r1_max) + " and " + str(r2_min) + " <= r2 <= " + str(r2_max))
    if r2_min is None and r2_max is None:
        r2 = None
    else:
        r2 = [r2_min, r2_max]

    return r1_min, r1_max, r2


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


def prep_label(r_min, r_max, r2):
    """
    Return label with r min and r max
    :param r2: None or range2 as list [x, y]
    """
    round_to = 1
    lsst_filt_label = "Target: " + \
        str(round(r_min, round_to)) + "≤ r ≤" + \
        str(round(r_max, round_to))
    if r2 is not None:
        lsst_filt_label += "\n and " + \
            str(round(r2[0], round_to)) + "≤ r ≤" + \
            str(round(r2[1], round_to))
    return lsst_filt_label


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

    min_lsst_val = lsst_df.iloc[:, 0].min()
    print("Min LSST value for " + str(lsst_feature_name) +
          " : " + str(min_lsst_val))
    max_lsst_val = lsst_df.iloc[:, 0].max()
    print("Max LSST value for " + str(lsst_feature_name) +
          " : " + str(max_lsst_val))

    # Set ranges of values to search over

    if thex_class_name == "Ia-91bg":
        # Ia-91bg r range: 16 - 26.8
        n = 20
        min_vals = np.linspace(min_lsst_val, max_lsst_val, n)
        max_vals = np.linspace(min_lsst_val, max_lsst_val, n)
    elif thex_class_name == "II":
        # II r range: 15.5 - 31.4
        min_vals = np.linspace(min_lsst_val, 19, 40)
        max_vals = np.linspace(16, 22, 60)  # [19.35, 17.13]
    elif thex_class_name == "TDE":
        # TDE r range: 16.6 - 30
        min_vals = [min_lsst_val]  # np.linspace(min_lsst_val, 20, 40)
        max_vals = np.linspace(21, 22, 30)
    elif thex_class_name == "Ia":
        # Ia r range: 14.3 - 29.7
        num_samples = 20
        min_vals1 = np.linspace(min_lsst_val, 14.4, num_samples)
        min_vals2 = np.linspace(24, 26, num_samples)

        max_vals2 = np.linspace(18, 20.6, num_samples)
        max_vals1 = np.linspace(25, 27, num_samples)

        min_vals = np.concatenate((min_vals1, min_vals2))
        max_vals = np.concatenate((max_vals1, max_vals2))

    else:
        min_vals = np.linspace(min_lsst_val, 16, 10)
        max_vals = np.linspace(15, 22, 10)

    delim = "-" * 100
    lsst_Z_orig = lsst_df["true_z"].values

    # All Features best params

    if len(thex_Z_AF) > 25:
        print(delim + "\nEstimating for all-features dataset")
        best_min_AF, best_max_AF, r2 = get_best_KS_double_range(lsst_df=lsst_df,
                                                                thex_redshifts=thex_Z_AF,
                                                                min_vals=min_vals,
                                                                max_vals=max_vals)
        lsst_data_AF = get_LSST_filt_redshifts(min_feature=best_min_AF,
                                               max_feature=best_max_AF,
                                               data=lsst_df,
                                               r2=r2)
        lsst_filt_label = prep_label(best_min_AF, best_max_AF, r2)

        plot_redshift_compare(thex_data=thex_Z_AF,
                              lsst_orig=lsst_Z_orig,
                              lsst_filt=lsst_data_AF,
                              lsst_filt_label=lsst_filt_label,
                              cname=thex_class_name,
                              dataset='allfeatures')

    # g-W2 dataset best params
    print(delim + "\nEstimating for g-W2-dataset")
    best_min_GW2, best_max_GW2, r2 = get_best_KS_double_range(lsst_df=lsst_df,
                                                              thex_redshifts=thex_Z_gw2,
                                                              min_vals=min_vals,
                                                              max_vals=max_vals)

    lsst_data_gw2 = get_LSST_filt_redshifts(min_feature=best_min_GW2,
                                            max_feature=best_max_GW2,
                                            data=lsst_df,
                                            r2=r2)

    lsst_filt_label = prep_label(best_min_GW2, best_max_GW2, r2)

    plot_redshift_compare(thex_data=thex_Z_gw2,
                          lsst_orig=lsst_Z_orig,
                          lsst_filt=lsst_data_gw2,
                          lsst_filt_label=lsst_filt_label,
                          cname=thex_class_name,
                          dataset='gW2')


if __name__ == "__main__":
    main(sys.argv)
