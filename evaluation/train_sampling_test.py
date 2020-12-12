"""
Determine if the model does any better or worse with a different distribution of transient events for the training data. Use 2 different training sets and same test set to see if training on LSST-like data results in better performance on LSST-like testing data, than by training on our normal training data and testing on the same LSST-like data. IE - the most 'LSST-like' data.
"""

import pandas as pd

from models.multi_model.multi_model import MultiModel
from evaluation.sampling_test import get_test_performance, get_test_sets
from evaluation.plotting import *


def main():
    cols = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
            "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag', 'redshift']

    Ia_label = 'I, Ia, _ROOT, _SN, _W_UVOPT, Unspecified Ia'
    II_label = 'CC, II, _ROOT, _SN, _W_UVOPT, Unspecified II'

    model = MultiModel(cols=cols,
                       num_runs=10,
                       class_labels=['Unspecified Ia', 'Unspecified II'],
                       transform_features=True,
                       min_class_size=40,
                       data_file="/Users/marina/Documents/PhD/research/astro_research/data/catalogs/v7/THEx-assembled-v7.1a-mags-legacy-xcalib-minxcal.fits"
                       )

    thex_dataset = pd.concat([model.X, model.y], axis=1)
    output_dir = "../figures/testing"
    lsst_X, lsst_y, rand_X, rand_y = get_test_sets(thex_dataset=thex_dataset,
                                                   output_dir=output_dir,
                                                   index="",
                                                   num_samples=400)

    # Drop redshift from X's

    if 'redshift' in list(lsst_X):
        print("\n\n Dropping redshift\n")
        lsst_X = lsst_X.drop(labels=['redshift'], axis=1)
        rand_X = rand_X.drop(labels=['redshift'], axis=1)

    LSST_results = []
    orig_results = []
    for i in range(model.num_runs):
        #################
        # Set up training set
        # Train on 80% 'normal' data, and test on remaining 20%
        # sample from rand X and y to get equal class counts as in test_LSST
        # compare 2 test set results
        train_size = int(rand_X.shape[0] * 0.8)
        train_X = rand_X.sample(train_size)
        train_y = rand_y.iloc[train_X.index]

        #################
        # Set up test sets
        # Rand testing is remaining data
        rand_test_X = rand_X.loc[~rand_X.index.isin(train_X.index)]
        rand_test_y = rand_y.iloc[rand_test_X.index].reset_index(drop=True)
        rand_test_X = rand_test_X.reset_index(drop=True)

        # Ensure LSST test set has same number of Ia and II
        target_counts = rand_test_y.groupby(['transient_type']).size()
        II_count = target_counts[II_label]
        Ia_count = target_counts[Ia_label]

        label_counts = {Ia_label: Ia_count, II_label: II_count}
        class_dfs_y = []
        class_dfs_X = []
        for class_label, class_count in label_counts.items():
            class_y = lsst_y.loc[lsst_y['transient_type']
                                 == class_label].sample(class_count)
            class_dfs_y.append(class_y)
            class_X = lsst_X.iloc[class_y.index]
            class_dfs_X.append(class_X)

        lsst_test_X = pd.concat(class_dfs_X).reset_index(drop=True)
        lsst_test_y = pd.concat(class_dfs_y).reset_index(drop=True)
        print("\n\nClass counts rand test set ")
        print(target_counts)
        print("\n\nClass counts LSST test set ")
        print(lsst_test_y.groupby(['transient_type']).size())

        #################
        # Train and test
        model.train_model(train_X.reset_index(drop=True),
                          train_y.reset_index(drop=True))

        LSST_results.append(get_test_performance(lsst_test_X, lsst_test_y, model))
        orig_results.append(get_test_performance(rand_test_X, rand_test_y, model))

    # Save attributes to model for performance measuremetns
    target_counts = lsst_test_y.groupby(['transient_type']).size()
    II_count = target_counts[II_label]
    Ia_count = target_counts[Ia_label]
    print("\noriginal model class counts ")
    print(model.class_counts)
    model.class_counts = [Ia_count, II_count]
    print('\nnew class counts')
    print(model.class_counts)

    plot_performance(model, lsst_test_y, "../figures/testing/test_LSST", LSST_results)
    plot_performance(model, rand_test_y, "../figures/testing/test_rand", orig_results)
    plot_performance_together(model, lsst_test_y, LSST_results, orig_results)


if __name__ == "__main__":
    main()
