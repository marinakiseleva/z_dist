# !/usr / bin / env python
# coding: utf-8
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
