"""
Determine if the model does any better or worse with a different distribution of transient events.
"""
import os
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    model.visualize_performance()


def get_test_performance(X, y, model):
    """
    Run model on this test set and return results
    """
    # Test model
    probabilities = model.get_all_class_probabilities(X, model.normalize)
    # Add labels as column to probabilities, for later evaluation
    label_column = y['transient_type'].values.reshape(-1, 1)
    probabilities = np.hstack((probabilities, label_column))
    return probabilities


def get_test_results(model, LSST_X_test, LSST_y_test, orig_X_test, orig_y_test, output_dir):
    """
    Train on model data and test on passed in data for X trials, and visualize results.
    """

    # model.dir = output_dir + "/training"
    # os.mkdir(model.dir)

    model.num_runs = 2
    model.num_folds = None
    LSST_results = []
    orig_results = []
    for i in range(model.num_runs):
        # Randomly sample 90% of training data for training
        X_train = model.X.sample(frac=0.9)
        y_train = model.y.iloc[X_train.index].reset_index(drop=True)
        X_train.reset_index(drop=True, inplace=True)

        # Ensure all X sets have columns in same order
        LSST_X_test = LSST_X_test[ordered_mags]
        orig_X_test = orig_X_test[ordered_mags]
        X_train = X_train[ordered_mags]

        # Train model on sampled set
        model.train_model(X_train, y_train)

        # Test model on LSST
        LSST_results.append(get_test_performance(LSST_X_test, LSST_y_test, model))

        # Test model on orig sample
        orig_results.append(get_test_performance(orig_X_test, orig_y_test, model))

    # Visualize performance of randomly sampled data
    plot_performance(model, orig_y_test, output_dir + "/orig_test", orig_results)

    # Visualize performance of LSST-like sampled data
    plot_performance(model, LSST_y_test, output_dir + "/lsst_test", LSST_results)


def get_THEx_sampled_data(class_name, max_rmag, num_samples, thex_dataset, output_dir=None):
    """
    Sample THEx class data to have the same redshift distribution as LSST cut to a certain r_first_mag
    :param thex_dataset: Dataframe of features and transient_type column
    """
    feature_name = "r_first_mag"

    thex_class_data = get_thex_class_data("Unspecified " + class_name, thex_dataset)

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
        if f_df.shape[0] > samples:
            f_df = f_df.sample(n=int(samples))
            new_data.append(f_df)
        else:
            new_data.append(f_df)
    df = pd.concat(new_data)
    df = df.reset_index(drop=True)
    if max_rmag is not None:
        lsst_label = "LSST (" + feature_name + " <= " + str(max_rmag) + ")"
    else:
        lsst_label = "LSST"
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
    if output_dir is not None:
        plt.savefig(output_dir + "/" + class_name)

    print("Class count : " + str(class_count))

    return df, random_sample


def main():

    # Initialize output directory
    exp = str(random.randint(1, 10**10))
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
    output_dir = ROOT_DIR + "/figures/evaluation/" + exp
    os.mkdir(output_dir)

    cols = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
            "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag',
            'redshift']
    model = MultiModel(cols=cols,
                       class_labels=['Unspecified Ia', 'Unspecified II'],
                       transform_features=False,
                       min_class_size=40
                       )
    model.dir = output_dir

    thex_dataset = pd.concat([model.X, model.y], axis=1)

    Ia_sampled, Ia_rand_sample = get_THEx_sampled_data(class_name="Ia",
                                                       max_rmag=None,
                                                       num_samples=200,
                                                       thex_dataset=thex_dataset,
                                                       output_dir=output_dir)
    II_sampled, II_rand_sample = get_THEx_sampled_data(class_name="II",
                                                       max_rmag=None,
                                                       num_samples=200,
                                                       thex_dataset=thex_dataset,
                                                       output_dir=output_dir)

    # f = 'r_mag'

    # plot_compare_feature_dists(feature_name=f,
    #                            class_name="Unspecified Ia",
    #                            rand_sample=Ia_rand_sample,
    #                            sampled=Ia_sampled)
    # plot_compare_feature_dists(feature_name=f,
    #                            class_name="Unspecified II",
    #                            rand_sample=II_rand_sample,
    #                            sampled=II_sampled)
    # plt.ioff()

    lsst_sampled_X, lsst_sampled_y = get_source_target(
        pd.concat([Ia_sampled, II_sampled]))

    orig_sampled_X, orig_sampled_y = get_source_target(
        pd.concat([Ia_rand_sample, II_rand_sample]))

    print("Original size of training set " + str(model.X.shape[0]))
    # Update training data to remove testing sets
    train_X, train_y = get_training_data(
        lsst_sampled_X, orig_sampled_X, model.X, model.y)
    print("New size of training set " + str(train_X.shape[0]))
    model.X = train_X[ordered_mags]
    model.y = train_y

    get_test_results(model=model,
                     LSST_X_test=lsst_sampled_X,
                     LSST_y_test=lsst_sampled_y,
                     orig_X_test=orig_sampled_X,
                     orig_y_test=orig_sampled_y,
                     output_dir=output_dir)


if __name__ == "__main__":
    main()
