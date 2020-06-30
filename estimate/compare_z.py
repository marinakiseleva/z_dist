# !/usr / bin / env python
# coding: utf-8

"""
Compare z dists of Rubin data and THEx data for passed-in class
"""


import sys
import numpy as np

from estimate.constants import *
from estimate.plotting import *
from estimate.get_data import *
from estimate.ks_matching import *


from models.multi_model.multi_model import MultiModel


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

    lsst_feature_name = "r_first_mag"

    cols = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
            "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag',
            'redshift']
    model = MultiModel(cols=cols,
                       class_labels=[thex_class_name],
                       transform_features=False,
                       min_class_size=40
                       )
    thex_data = model.X['redshift'].values

    LSST_df = get_lsst_class_data(class_name=lsst_class_name,
                                  feature_name=lsst_feature_name)

    file_title = "z_compare_" + lsst_class_name + ".png"
    datas = {"THEx": thex_data,
             "LSST_orig": LSST_df[LSST_Z_COL]}
    plot_Z_ranges(lsst_class_name, datas, file_title)


if __name__ == "__main__":
    main(sys.argv)
