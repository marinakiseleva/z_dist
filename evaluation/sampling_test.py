"""
Determine if the model does better or worse with a different distribution of Ia events, that more strongly resemble the PLASTICCS catalog filtered to a certain range (Ia events with initial r mag <=23 )
"""

from estimate.get_data import *
import numpy as np
import matplotlib.pyplot as plt

import math


def plot_z_dists(data1, data2, label1, label2, bins, class_name):
    """
    Plot z distribution of two sets
    """
    fig, ax = plt.subplots(tight_layout=True, sharex=True,  sharey=True)
    b = ax.hist(data1, density=True, bins=bins,
                label=label1, fill=False, edgecolor='blue')
    a = ax.hist(data2, density=True, bins=bins,
                label=label2, fill=False, edgecolor='red')
    plt.legend()
    plt.title(class_name)
    plt.xlabel("Redshift")
    plt.ylabel("Density")
    plt.savefig("../figures/evaluation/z_dist_" + class_name)


def get_THEx_sampled_data(class_name, max_rmag, num_samples):
    """
    Sample THEx class data to have the same redshift distribution as LSST cut to a certain r_first_mag
    """
    feature_name = "r_first_mag"

    # Pull down THEx data
    thex_dataset = get_data('g_W2')
    thex_class_data = get_thex_class_data(class_name, thex_dataset)

    thex_z_vals = thex_class_data['redshift'].values

    # Pull down LSST data
    lsst_class_data = get_lsst_class_data(class_name, feature_name)

    # Get histogram and per bar frequencies of LSST cut data
    # 1. get LSST data cut to certain r first mag range
    filt_lsst_class_data = lsst_class_data[lsst_class_data[feature_name] <= max_rmag]

    # 2. Get hist of redshift values, and frequencies
    lsst_z_vals = filt_lsst_class_data['true_z'].values
    Z_bins = np.linspace(0, 1, 50)
    hist, bins = np.histogram(lsst_z_vals, bins=Z_bins)
    z_dist = hist / len(lsst_z_vals)  # proportion of total in each bin

    # Sample THEx data at these rates
    new_data = []
    for index, freq in enumerate(z_dist):
        samples = num_samples * freq
        min_feature = Z_bins[index]
        max_feature = Z_bins[index + 1]
        # Filter by redshift
        f_df = thex_class_data[(thex_class_data['redshift'] >= min_feature) & (
            thex_class_data['redshift'] <= max_feature)]
        if f_df.shape[0] >= samples:
            f_df = f_df.sample(n=int(samples))
            new_data.append(f_df)
        else:
            new_data.append(f_df)
    df = pd.concat(new_data)
    df = df.reset_index(drop=True)
    lsst_label = "LSST, " + feature_name + " <= " + str(max_rmag)
    thex_label = "THEx, sampled"
    plot_z_dists(lsst_z_vals, df['redshift'].values,
                 lsst_label, thex_label, Z_bins, class_name)

    return df
