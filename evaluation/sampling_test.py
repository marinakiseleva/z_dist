"""
Determine if the model does any better or worse with a different distribution 
of transient events, for the testing data. 
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from datetime import datetime

from thex_data.data_consts import *
from models.multi_model.multi_model import MultiModel

from evaluation.plotting import *
from estimate.get_data import *

ordered_mags = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
                "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag']


def remove_data(alt_X, orig_X, orig_y):
    """
    Keep only data in orig_X that does not appear in alt_X
    """

    orig_X = orig_X[ordered_mags].astype('float32')
    combined_orig = pd.concat([orig_X, orig_y], axis=1)

    # Reorder columns so they are consistent
    alt_X = alt_X[ordered_mags].astype('float32')
    # combined_orig = combined_orig[ordered_mags].astype('float32')

    # Keep rows that only exist in left
    merged = pd.merge(left=combined_orig,
                      right=alt_X,
                      indicator=True,
                      how='outer',
                      on=ordered_mags)
    merged = merged.loc[merged['_merge'] == 'left_only']
    merged = merged.drop('_merge', axis=1).reset_index(drop=True)

    new_y = merged[[TARGET_LABEL]]
    new_X = merged.drop([TARGET_LABEL], axis=1)
    return new_X, new_y


def get_training_data(lsst_test_X, THEx_test_X, all_X, all_y):
    """
    Return sampled X and y, with NO values in 'avoid' DataFrame
    :param lsst_test_X: DataFrame of data sampled like LSST
    :param THEx_test_X: DataFrame of data sampled like our data
    :param all_X: All X data in initialized model
    :param all_y: All y data in initialized model
    """
    X_train = all_X.copy(True)
    y_train = all_y.copy(True)

    # Drop LSST testing data from training
    reduced_X, reduced_y = remove_data(alt_X=lsst_test_X.copy(True),
                                       orig_X=X_train,
                                       orig_y=y_train)
    # Drop THEx testing data from training
    X_train_reduced, y_train_reduced = remove_data(alt_X=THEx_test_X.copy(True),
                                                   orig_X=reduced_X,
                                                   orig_y=reduced_y)
    X_train_reduced = X_train_reduced.reset_index(drop=True)
    y_train_reduced = y_train_reduced.reset_index(drop=True)
    return X_train_reduced, y_train_reduced


def get_source_target(data):
    """
    Splits DataFrame into X and y
    :param df: Original DataFrame
    """
    sampled_data = data.sample(frac=1).reset_index(drop=True)
    sampled_y = sampled_data[[TARGET_LABEL]]
    sampled_X = sampled_data.drop(labels=[TARGET_LABEL], axis=1)
    return sampled_X, sampled_y


def get_test_performance(X, y, model):
    """
    Run model on this test set and return results
    """
    # Test model
    probabilities = model.get_all_class_probabilities(X)
    # Add labels as column to probabilities, for later evaluation
    label_column = y[TARGET_LABEL].values.reshape(-1, 1)
    probabilities = np.hstack((probabilities, label_column))
    return probabilities


def get_test_sets(thex_dataset, output_dir, index):
    """
    Return X and y of LSST and random sampled testing sets.
    Sample the given numbers of Ia and II to get roughly 100 samples each.
    """

    lsst_df = get_lsst_data()
    Ia_sampled, Ia_rand_sample, Ia_LSST_Z = get_THEx_sampled_data(lsst_df=lsst_df,
                                                                  class_name="Unspecified Ia",
                                                                  num_samples=110,
                                                                  thex_dataset=thex_dataset)
    II_sampled, II_rand_sample, II_LSST_Z = get_THEx_sampled_data(lsst_df=lsst_df,
                                                                  class_name="II",
                                                                  num_samples=110,
                                                                  thex_dataset=thex_dataset)

    Iacount = Ia_sampled.shape[0]
    IIcount = II_sampled.shape[0]
    if Iacount < IIcount:
        II_sampled = II_sampled.sample(Iacount).reset_index(drop=True)
        II_rand_sample = II_rand_sample.sample(Iacount).reset_index(drop=True)

    plot_sample_dists_together(Ia_sampled,
                               Ia_rand_sample,
                               Ia_LSST_Z,
                               II_sampled,
                               II_rand_sample,
                               II_LSST_Z, output_dir)

    lsst_sampled_X, lsst_sampled_y = get_source_target(
        pd.concat([Ia_sampled, II_sampled]))

    orig_sampled_X, orig_sampled_y = get_source_target(
        pd.concat([Ia_rand_sample, II_rand_sample]))
    return lsst_sampled_X, lsst_sampled_y, orig_sampled_X, orig_sampled_y


def get_THEx_sampled_data(lsst_df, class_name, num_samples, thex_dataset):
    """
    Create 2 sample test sets from THEx data, one randomly sampled from our data and the other sampled with LSST redshift dist
    :param thex_dataset: DataFrame of THEx data, X and y 
    """
    print("\nSampling Class: " + class_name)
    thex_class_data = get_thex_class_data(class_name, thex_dataset)
    lsst_class_name = class_name.replace("Unspecified ", "")
    lsst_z_vals = get_lsst_class_Zs(class_name=lsst_class_name,
                                    lsst_df=lsst_df)

    # Get hist of redshift values, and frequencies
    Z_bins = np.linspace(0, 1, 50)
    hist, bins = np.histogram(lsst_z_vals, bins=Z_bins)
    z_dist = hist / len(lsst_z_vals)  # proportion of total in each bin

    # Create LSST sample by sampling THEx data at LSST z rates
    lsst_sample = []
    for index, freq in enumerate(z_dist):
        samples = int(round(freq, 2) * num_samples)
        # samples = int(num_samples * freq)
        min_feature = Z_bins[index]
        max_feature = Z_bins[index + 1]
        # Filter by redshift
        f_df = thex_class_data[(thex_class_data[Z_FEAT] >= min_feature) & (
            thex_class_data[Z_FEAT] <= max_feature)]
        if samples == 0:
            continue
        elif f_df.shape[0] > samples:
            f_df = f_df.sample(n=samples)
            lsst_sample.append(f_df)
        else:
            missing = samples - f_df.shape[0]
            # print("Not enough data in range " + str(index) + " by " + str(missing))
            lsst_sample.append(f_df)
    lsst_sample = pd.concat(lsst_sample).reset_index(drop=True)

    class_count = lsst_sample.shape[0]
    random_sample = thex_class_data.sample(class_count).reset_index(drop=True)

    return lsst_sample, random_sample, lsst_z_vals


def get_cc(y, cn):
    """
    get class counts
    """
    a = 0
    for index, row in y.iterrows():
        labels = convert_str_to_list(row[TARGET_LABEL])
        if cn in labels:
            a += 1
    return a


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
        print("\n\nIteration " + str(i + 1) + "/" + str(model.num_runs))
        # Resample testing sets each run
        X_lsst, y_lsst, X_orig, y_orig = get_test_sets(
            thex_dataset, output_dir, i)

        # Update training data to remove testing sets
        X_train, y_train = get_training_data(X_lsst, X_orig, model.X, model.y)

        # Ensure all X sets have columns in same order
        X_lsst = X_lsst[ordered_mags]
        X_orig = X_orig[ordered_mags]
        X_train = X_train[ordered_mags]

        print("\nTraining set size: " + str(X_train.shape[0]))
        for c in model.class_labels:
            print(c + " test count, LSST: " + str(get_cc(y_lsst, c)) +
                  ", THEx: " + str(get_cc(y_orig, c)))

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

    plot_performance_together(model, y_lsst, LSST_results, orig_results, output_dir)


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
            Z_FEAT]

    codes = ["A1", "F1", "B1", "G1"]
    model = MultiModel(cols=cols,
                       class_labels=['Unspecified Ia', 'II'],
                       transform_features=False,
                       case_code=codes,
                       min_class_size=40,
                       data_file=CUR_DATA_PATH,
                       )
    model.dir = output_dir

    get_test_results(model=model,
                     output_dir=output_dir,
                     iterations=10)


if __name__ == "__main__":
    main()
