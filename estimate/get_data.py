# !/usr / bin / env python
# coding: utf-8
import pandas as pd
import numpy as np
import pickle
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

def get_lsst_data():
    """
    Pull down LSST data
    """
    with open(DATA_DIR + 'lsst-sims.pk', 'rb') as f:
        data = pickle.load(f)
    return data


def get_lsst_class_data(class_name, feature_name):
    """
    Filter LSST data to only those samples with this class name, and valid values for feature name. Return as Pandas DataFrame with first column as feature values and second column as z
    """
    data = get_lsst_data()
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
