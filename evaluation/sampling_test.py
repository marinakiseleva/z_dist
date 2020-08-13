"""
Determine if the model does any better or worse with a different distribution of transient events, for the testing data. 
"""
from datetime import datetime
import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from thex_data.data_consts import *
from models.multi_model.multi_model import MultiModel
from evaluation.plotting import *
from estimate.get_data import *

ordered_mags = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
                "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag']


def remove_data(alt_X, orig_X, orig_y):
    """
    Drop data from orig_X that is also in alt_X
    """
    # Reorder columns so they are consistent
    alt_X = alt_X[ordered_mags].astype('float32')
    orig_X = orig_X[ordered_mags].astype('float32')

    new_df = orig_X.merge(right=alt_X, how='inner',
                          on=ordered_mags, right_index=True)
    drop_indices = new_df.index.tolist()

    # Drop testing data from training
    new_X = orig_X.drop(index=drop_indices).reset_index(drop=True)
    new_y = orig_y.drop(index=drop_indices).reset_index(drop=True)
    return new_X, new_y


def get_training_data(lsst_sampled_X, orig_sampled_X, all_X, all_y):
    """
    Return sampled X and y, with NO values in 'avoid' DataFrame
    :param lsst_sampled_data: DataFrame of data sampled like LSST
    :param orig_sampled_data: DataFrame of data sampled like our data
    :param all_X: All X data in initialized model
    :param all_y: All y data in initialized model
    """
    X_train = all_X.copy()
    y_train = all_y.copy()

    # Drop LSST testing data from training
    reduced_X, reduced_y = remove_data(alt_X=lsst_sampled_X,
                                       orig_X=X_train,
                                       orig_y=y_train)
    # Drop our testing data from training
    new_X, new_y = remove_data(alt_X=orig_sampled_X,
                               orig_X=reduced_X,
                               orig_y=reduced_y)

    return new_X, new_y


def get_source_target(data):
    """
    Splits DataFrame into X and y
    :param df: Original DataFrame
    """
    sampled_data = data.sample(frac=1).reset_index(drop=True)
    sampled_y = sampled_data[['transient_type']]
    sampled_X = sampled_data.drop(labels=['transient_type'], axis=1)
    return sampled_X, sampled_y


def plot_performance(model, testdata_y, output_dir, results):
    """
    """
    # Reset class counts and y to be that of the test set, so baselines are accurate
    a = testdata_y.groupby('transient_type').size()[
        'I, Ia, _ROOT, _SN, _W_UVOPT, Unspecified Ia']
    b = testdata_y.groupby('transient_type').size()[
        'CC, II, _ROOT, _SN, _W_UVOPT, Unspecified II']
    model.class_counts = {"Unspecified Ia": a,
                          "Unspecified II": b}

    model.results = results
    model.y = testdata_y
    model.dir = output_dir

    os.mkdir(model.dir)

    # Save results in pickle
    with open(model.dir + '/results.pickle', 'wb') as f:
        pickle.dump(model.results, f)
    with open(model.dir + '/y.pickle', 'wb') as f:
        pickle.dump(model.y, f)

    model.visualize_performance()


def get_test_performance(X, y, model):
    """
    Run model on this test set and return results
    """
    # Test model
    probabilities = model.get_all_class_probabilities(X)
    # Add labels as column to probabilities, for later evaluation
    label_column = y['transient_type'].values.reshape(-1, 1)
    probabilities = np.hstack((probabilities, label_column))
    return probabilities


def get_test_sets(thex_dataset, output_dir, index, num_samples=200):
    """
    Return X and y of LSST and random sampled testing sets
    """
    Ia_sampled, Ia_rand_sample = get_THEx_sampled_data(class_name="Ia",
                                                       num_samples=num_samples,
                                                       thex_dataset=thex_dataset,
                                                       output_dir=output_dir,
                                                       i=index)
    II_sampled, II_rand_sample = get_THEx_sampled_data(class_name="II",
                                                       num_samples=num_samples,
                                                       thex_dataset=thex_dataset,
                                                       output_dir=output_dir,
                                                       i=index)

    lsst_sampled_X, lsst_sampled_y = get_source_target(
        pd.concat([Ia_sampled, II_sampled]))

    orig_sampled_X, orig_sampled_y = get_source_target(
        pd.concat([Ia_rand_sample, II_rand_sample]))
    return lsst_sampled_X, lsst_sampled_y, orig_sampled_X, orig_sampled_y


def get_THEx_sampled_data(class_name, num_samples, thex_dataset, output_dir, i="", max_rmag=None):
    """
    Create 2 sample test sets from THEx data, one randomly sampled from our data and the other sampled with LSST redshift dist
    :param thex_dataset: DataFrame of THEx data, X and y
    :param i: iteration 
    """

    thex_class_data = get_thex_class_data("Unspecified " + class_name, thex_dataset)

    # Pull down LSST data
    feature_name = "r_first_mag"
    lsst_class_data = get_lsst_class_data(class_name, feature_name)
    # cut LSST data to r mag if needed
    lsst_label = "Rubin"
    if max_rmag is not None:
        lsst_class_data = lsst_class_data[lsst_class_data[feature_name] <= max_rmag]
        lsst_label += " (" + feature_name + " <= " + str(max_rmag) + ")"

    # 2. Get hist of redshift values, and frequencies
    lsst_z_vals = lsst_class_data['true_z'].values
    Z_bins = np.linspace(0, 1, 50)
    hist, bins = np.histogram(lsst_z_vals, bins=Z_bins)
    z_dist = hist / len(lsst_z_vals)  # proportion of total in each bin

    # Create LSST sample by sampling THEx data at LSST z rates
    lsst_sample = []
    for index, freq in enumerate(z_dist):
        samples = num_samples * freq
        min_feature = Z_bins[index]
        max_feature = Z_bins[index + 1]
        # Filter by redshift
        f_df = thex_class_data[(thex_class_data['redshift'] >= min_feature) & (
            thex_class_data['redshift'] <= max_feature)]
        if f_df.shape[0] > samples:
            f_df = f_df.sample(n=int(samples))
            lsst_sample.append(f_df)
        else:
            lsst_sample.append(f_df)
    lsst_sample = pd.concat(lsst_sample).reset_index(drop=True)

    class_count = lsst_sample.shape[0]
    random_sample = thex_class_data.sample(class_count).reset_index(drop=True)

    # Plot LSST data, sampled LSST, and random sample
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                           dpi=DPI,
                           tight_layout=True, sharex=True,  sharey=True)

    GREEN = "#b3e6b3"
    BLUE = "#99c2ff"
    RED = "#ffb3b3"

    a = ax.hist(lsst_z_vals, density=True, bins=Z_bins,
                label=lsst_label, fill=False, edgecolor=BLUE, linewidth=1.2)
    b = ax.hist(random_sample['redshift'].values, density=True, bins=Z_bins,
                label="THEx random sample", fill=False, edgecolor=GREEN, linewidth=1.2)
    c = ax.hist(lsst_sample['redshift'].values, density=True, bins=Z_bins,
                label="THEx Rubin sample", fill=False, edgecolor=RED, linewidth=1.2)
    plt.xticks(fontsize=TICK_S)
    plt.yticks(fontsize=TICK_S)
    plt.legend(fontsize=LAB_S)
    plt.title(class_name + " (unspec.)", fontsize=TITLE_S)
    plt.xlabel("Redshift", fontsize=LAB_S)
    plt.ylabel("Density", fontsize=LAB_S)
    plt.savefig(output_dir + "/" + class_name + "_" + str(i) + ".pdf")
    return lsst_sample, random_sample


def get_test_results(model, output_dir, iterations=100):
    """
    Train on model data and test on passed in data for X trials, and visualize results.
    """
    model.num_runs = iterations
    model.num_folds = None
    thex_dataset = pd.concat([model.X, model.y], axis=1)

    LSST_results = []
    orig_results = []

    for i in range(model.num_runs):
        # Resample testing sets each run
        X_lsst, y_lsst, X_orig, y_orig = get_test_sets(
            thex_dataset, output_dir, i)

        # Update training data to remove testing sets
        X_train, y_train = get_training_data(X_lsst, X_orig, model.X, model.y)
        print("Training set size is " + str(X_train.shape[0]))

        # Ensure all X sets have columns in same order
        X_lsst = X_lsst[ordered_mags]
        X_orig = X_orig[ordered_mags]
        X_train = X_train[ordered_mags]

        # Train model on sampled set
        model.train_model(X_train, y_train)

        # Test model on LSST
        LSST_results.append(get_test_performance(X_lsst, y_lsst, model))

        # Test model on orig sample
        orig_results.append(get_test_performance(X_orig, y_orig, model))

    # Visualize performance of LSST-like sampled data
    plot_performance(model, y_lsst, output_dir + "/lsst_test", LSST_results)
    # Visualize performance of randomly sampled data
    plot_performance(model, y_orig, output_dir + "/orig_test", orig_results)


def main():

    # Initialize output directory

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
    output_dir = ROOT_DIR + "/output/" + dt_string
    os.mkdir(output_dir)

    init_plot_settings()

    cols = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
            "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag',
            'redshift']
    model = MultiModel(cols=cols,
                       class_labels=['Unspecified Ia', 'Unspecified II'],
                       transform_features=False,
                       min_class_size=40
                       )
    model.dir = output_dir

    get_test_results(model=model,
                     output_dir=output_dir)


if __name__ == "__main__":
    main()
