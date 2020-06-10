# !/usr / bin / env python
# coding: utf-8

"""
Match LSST and THEx redshift ranges (not distributions). So that the minimum and maximum redshift per class is roughly the same for THEx and LSST. Try to do so using one r_first_mag cut on LSST data. 
"""


import sys
import numpy as np

from estimate.constants import *
from estimate.plotting import *
from estimate.get_data import *
from estimate.ks_matching import *

LSST_Z_COL = 'true_z'
LSST_FEAT_COL = 'r_first_mag'


def get_filt_LSST_Z(min_feature, max_feature, data):
    """
    Get redshift values of LSST data with feature values in this range 
    :param min_feature: Minimum acceptable value of feature
    :param max_feature: Maximum acceptable value of feature
    :param data: LSST data as DataFrame with feature and redshift column
    """
    if min_feature is None:
        f_df = data[data[LSST_FEAT_COL] <= max_feature]
    elif max_feature is None:
        f_df = data[data[LSST_FEAT_COL] >= min_feature]
    else:
        f_df = data[(data[LSST_FEAT_COL] >= min_feature) &
                    (data[LSST_FEAT_COL] <= max_feature)]
    return f_df[LSST_Z_COL].values


def main(argv):
    """
    Call like python estimate_dists.py Ia Ia r
    Only features to choose from are g, r, i, z, y
    """
    # Initialize passed-in args
    thex_class_name = argv[1]
    lsst_class_name = argv[2]
    max_feat_val = argv[3]

    if len(argv) == 5:
        dataset = argv[4]
    else:
        dataset = 'g-W2'

    lsst_feature_name = "r_first_mag"

    LSST_df = get_lsst_class_data(class_name=lsst_class_name,
                                  feature_name=lsst_feature_name)

    # Pull down our data
    thex_Z_AF, thex_Z_gw2 = get_thex_z_data(thex_class_name)
    # default to g-w2 unless all-features is passed in
    if dataset == 'all-features':
        thex_data = thex_Z_AF
    else:
        thex_data = thex_Z_gw2

    min_feat_val = None
    # max_feat_val = 23
    file_title = "range_matching_" + lsst_class_name + ".png"
    lsst_Z_ranged = get_filt_LSST_Z(
        min_feature=None, max_feature=max_feat_val, data=LSST_df)

    prop_kept = round((len(lsst_Z_ranged) / LSST_df.shape[0]) * 100, 1)
    LSST_label = "Filtered LSST: " + \
        "r mag \N{LESS-THAN OR EQUAL TO}" + \
        str(max_feat_val) + ", " + str(prop_kept) + "%"

    datas = {"THEx": thex_data,
             "LSST_orig": LSST_df[LSST_Z_COL],
             "LSST_filt": lsst_Z_ranged}

    file_title = "range_matching_" + dataset + "_" + lsst_class_name + ".png"
    plot_Z_ranges(lsst_class_name, datas, LSST_label, file_title)

    return prop_kept

if __name__ == "__main__":
    main(sys.argv)
