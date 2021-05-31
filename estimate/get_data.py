# !/usr / bin / env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
import pickle
from estimate.constants import *
from thex_data.data_consts import *


def get_data(name):
    """
    Pull down project data, one of two types:
    all-features-dataset: 'all_features'
    g_W2-dataset: 'g_W2'
    """
    file = DATA_DIR + 'model_data_' + name + '.csv'
    data = pd.read_csv(file)
    data.drop(['Unnamed: 0'], axis=1, inplace=True)
    return data


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


def get_lsst_data():
    """
    Pull down LSST data
    """ 
    with open(DATA_DIR + 'LSST_data.pickle', 'rb') as f:
        return pickle.load(f) 


def get_lsst_class_Zs(class_name,  lsst_df):
    """
    Filter LSST data to only those samples with this class name. Return Zs for this class.
    """
    class_ids_names = {'Ia': 90, 'II': 42, 'Ibc': 62, 'Ia-91bg': 67, 'TDE': 15}

    targ_id = class_ids_names[class_name]
    return lsst_df.loc[lsst_df['true_target'] == targ_id]['true_z'].values

