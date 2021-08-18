# !/usr / bin / env python
# coding: utf-8

"""
Compare z dists of LSST data and THEx data for the 5 classes 
for which we have data for both (Ia, II, Ibc, TDE, Ia-91bg)
"""

import os
import sys
import numpy as np
import pickle
from estimate.constants import *
from estimate.plotting import *
from estimate.get_data import *
from os import path

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
    Plots redshift distribution of LSST data (from PLASTICC) vs THEx data, after initializing model, so it is only valid data. And we only use valid r-mag data of LSST.
    """
    # Only features to choose from are g, r, i, z, y
    lsst_feature_name = "tflux_r"

    cols = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
            "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag',
            Z_FEAT]
    # if path.exists('../data/zdata.pickle'):
    #     with open('../data/zdata.pickle', 'rb') as handle:
    #         data = pickle.load(handle)
    #         init_plot_settings()
    #         plot_Z_ranges_together(data, "z_compare_full.pdf")
    #         return 1
    model = MultiModel(cols=cols,
                       transform_features=False,
                       min_class_size=3,
                       data_file=CUR_DATA_PATH,
                       case_code=["A1", "F1", "B1", "G1"],
                       lsst_test=True,
                       identified_only=True
                       )

    data = {}
    lsst_df = get_lsst_data()
    for class_name in model.class_labels:
        class_indices = []
        for index, row in model.y.iterrows():
            if class_name in row['transient_type']:
                class_indices.append(index)
        thex_Z = model.X.iloc[class_indices][Z_FEAT].values
        LSST_Z = get_lsst_class_Zs(class_name=class_mapping[class_name],
                                   lsst_df=lsst_df)
        data[class_name] = {"THEx": thex_Z,
                            "LSST_orig": LSST_Z}

    # Save data
    with open(DATA_DIR + 'zdata.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    init_plot_settings()
    plot_Z_ranges_together(data, "z_compare_full.pdf")


if __name__ == "__main__":
    main(sys.argv)
