# !/usr / bin / env python
# coding: utf-8

"""
Compare z dists of LSST data and THEx data for the 4 classes for which we have data for both (Ia, II, TDE, Ia-91bg)
"""


import sys
import numpy as np

from estimate.constants import *
from estimate.plotting import *
from estimate.get_data import *


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


# Map from thex names to LSST names
class_mapping = {"Unspecified Ia": "Ia",
                 "Unspecified II": "II",
                 "TDE": "TDE",
                 "Ia-91bg": "Ia-91bg"}


def main(argv):
    """
    Plots redshift distribution of LSST data (from PLASTICC) vs THEx data, after initializing model, so it is only valid data. And we only use valid r-mag data of LSST.
    """
    # Only features to choose from are g, r, i, z, y
    lsst_feature_name = "r_first_mag"

    cols = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
            "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag',
            'redshift']
    model = MultiModel(cols=cols,
                       class_labels=["Unspecified Ia",
                                     "Unspecified II", "Ia-91bg", "TDE"],
                       transform_features=False,
                       min_class_size=40,
                       data_file="/Users/marina/Documents/PhD/research/astro_research/data/catalogs/v7/THEx-assembled-v7.1a-mags-legacy-xcalib-minxcal.fits"
                       )

    data = {}
    for class_name in model.class_labels:
        class_indices = []
        for index, row in model.y.iterrows():
            if class_name in row['transient_type']:
                class_indices.append(index)
        thex_Z = model.X.iloc[class_indices]['redshift'].values
        print("\n\nFor class name THEX values are ")
        print(thex_Z)
        LSST_Z = get_lsst_class_data(class_name=class_mapping[class_name],
                                     feature_name=lsst_feature_name)[LSST_Z_COL].values
        print("\n\nFor lsst ")
        print(LSST_Z)
        data[class_name] = {"THEx": thex_Z,
                            "LSST_orig": LSST_Z}

    init_plot_settings()
    plot_Z_ranges_together(data, "z_compare_full.pdf")


if __name__ == "__main__":
    main(sys.argv)
