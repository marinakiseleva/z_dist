import sys
from functools import partial
import multiprocessing

import pickle
import numpy as np
from scipy import stats
import math

from estimate.constants import *


def get_LSST_filt_redshifts(min_feature, max_feature, data, r2=None):
    """
    Get redshift values as Numpy array, for feature values in the range [min_feature, max_feature]. First column is features.
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

    passed = False
    if KS_statistic <= D_critical:
        passed = True

    # Return KS stat, p-value for
    return [KS_statistic, p_value, passed]


def get_best_range_index(ranges, stats, p_values, accepted, lsst_data, r2s=None):
    """
    Get index of best range (with most samples) that meets critical acceptance from KS test (KS_statistic < D_critical). 
    """
    max_count = 0
    best_r_index = None    # Index corresponding to largest true range
    for index, a in enumerate(accepted):
        min_val, max_val = ranges[index]
        r2_range = None
        if r2s is not None:
            min_val2, max_val2 = r2s[index]
            # Values are None when using no second range
            if min_val2 is not None and max_val2 is not None:
                r2_range = [min_val2, max_val2]

        sample_count = get_count(lsst_data, min_val, max_val, r2_range)
        if a and sample_count >= max_count:
            max_count = sample_count
            best_r_index = index

    return best_r_index


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


def get_best_range(ranges, stats, p_values, accepted, lsst_data, r2s=None):
    """
    Get best range  
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
    print("Current range " + str(index + 1) + "/" + str(len(ranges)), flush=True)
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


def get_best_KS_range(lsst_df, thex_redshifts, min_vals, max_vals):
    """
    Find min and max range for feature that gets LSST and THEx redshift distributions for this class as close together.
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
    Same as get_best_KS_range but can use 2 ranges.
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
