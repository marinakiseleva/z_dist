"""
Determine if the model does better or worse with a different distribution of Ia events, that more strongly resemble the PLASTICCS catalog filtered to a certain range (Ia events with initial r mag <=23 )
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from estimate.get_data import *


def remove_testing_data(X, y,  avoid):
    """
    Return sampled X and y, with NO values in 'avoid' DataFrame
    :param X: X data to sample from
    :param y: y data
    :param proportion: decimal proportion to sample
    :param avoid: X DataFrame of rows to NOT sample (they are used in testing)
    """
    keep_indices = []
    for index1, row in X.iterrows():
        for index2, row2 in avoid.iterrows():
            if not np.equal(row.values.astype('float32'), row2.values.astype('float32')).all():
                keep_indices.append(index1)
                break

    valid_X = X.loc[keep_indices, :].reset_index(drop=True)
    valid_y = y.loc[keep_indices, :].reset_index(drop=True)

    return valid_X, valid_y


def get_test_results(model, X_test, y_test):
    """
    Train on model data and test on passed in data for 10 trials, and visualize results.
    """
    NUM_TRIALS = 10
    TRAIN_PROP = .9
    results = []
    for i in range(NUM_TRIALS):
        X_train, y_train, X_test, y_test = model.manually_stratify(
            model.X, model.y, TRAIN_PROP)
        # TRAIN ON DATA NOT IN TEST!

        model.train_model(X_train, y_train)

        # Test model
        probabilities = model.get_all_class_probabilities(X_test, model.normalize)
        # Add labels as column to probabilities, for later evaluation
        label_column = y_test['transient_type'].values.reshape(-1, 1)
        probabilities = np.hstack((probabilities, label_column))
        results.append(probabilities)
    model.results = results
    model.visualize_performance()


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
    if max_rmag is not None:
        filt_lsst_class_data = lsst_class_data[lsst_class_data[feature_name] <= max_rmag]
    else:
        filt_lsst_class_data = lsst_class_data

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
    lsst_label = "LSST (" + feature_name + " <= " + str(max_rmag) + ")"
    thex_label = "THEx (LSST sample)"

    class_count = df.shape[0]
    random_sample = thex_class_data.sample(class_count).reset_index(drop=True)

    fig, ax = plt.subplots(tight_layout=True, sharex=True,  sharey=True)
    a = ax.hist(lsst_z_vals, density=True, bins=Z_bins,
                label=lsst_label, fill=False, edgecolor='blue')
    b = ax.hist(random_sample['redshift'].values, density=True, bins=Z_bins,
                label="THEx (random sample)", fill=False, edgecolor='green')
    c = ax.hist(df['redshift'].values, density=True, bins=Z_bins,
                label=thex_label, fill=False, edgecolor='red')

    plt.legend()
    plt.title(class_name)
    plt.xlabel("Redshift")
    plt.ylabel("Density")
    plt.savefig("../figures/evaluation/z_dist_" + class_name)

    print("Class count : " + str(class_count))

    return df, random_sample
