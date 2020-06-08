# !/usr / bin / env python
# coding: utf-8

"""
Try to match LSST and THEx redshift distributions using KS 2-sided-test.
"""


import sys
import pickle
import numpy as np

from estimate.constants import *
from estimate.plotting import *
from estimate.get_data import *
from estimate.ks_matching import *


def prep_label(r_min, r_max, min_lsst_val, r2):
    """
    Return label with r min and r max
    :param r2: None or range2 as list [x, y]
    """
    round_to = 1
    lsst_filt_label = "Target: "
    if r_min != min_lsst_val:
        lsst_filt_label += str(round(r_min, round_to)) + "≤ "

    lsst_filt_label += "r ≤" + str(round(r_max, round_to))
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
        n = 80
        min_vals = np.linspace(min_lsst_val, 16.1, n)
        max_vals = np.linspace(18, 20, n)
    elif thex_class_name == "II":
        # II r range: 15.5 - 31.4
        # min_vals = np.array([min_lsst_val])
        # max_vals = np.array([19.31812749881689, 17.07244148433803, 19.5])
        min1 = np.array([min_lsst_val])
        max1 = np.linspace(18, 20, 400)
        min2 = np.linspace(18, 20, 400)
        max2 = np.linspace(20, 22, 400)
        min_vals = np.concatenate((min1, min2))
        max_vals = np.concatenate((max1, max2))
    elif thex_class_name == "TDE":
        # TDE r range: 16.6 - 30
        min_vals = np.array([min_lsst_val])
        max_vals = np.linspace(21, 22, 80)
    elif thex_class_name == "Ia":
        # Ia r range: 14.3 - 29.7
        # min_vals = np.array([min_lsst_val])
        min_vals = np.concatenate((np.array([min_lsst_val]), np.linspace(21, 22, 200)))
        # max_vals = np.linspace(15, 17, 90)
        max_vals = np.concatenate((np.linspace(19, 20, 100), np.linspace(21, 22, 200)))

    else:
        n = 40
        min_vals = np.linspace(min_lsst_val, max_lsst_val, n)
        max_vals = np.linspace(min_lsst_val, max_lsst_val, n)

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
        lsst_filt_label = prep_label(best_min_AF, best_max_AF, min_lsst_val, r2)

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

    lsst_filt_label = prep_label(best_min_GW2, best_max_GW2, min_lsst_val, r2)

    plot_redshift_compare(thex_data=thex_Z_gw2,
                          lsst_orig=lsst_Z_orig,
                          lsst_filt=lsst_data_gw2,
                          lsst_filt_label=lsst_filt_label,
                          cname=thex_class_name,
                          dataset='gW2')


if __name__ == "__main__":
    main(sys.argv)
