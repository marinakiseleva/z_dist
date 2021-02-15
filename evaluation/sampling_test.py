"""
Use same training, 2 different testing sets.
"""

import pandas as pd
import pickle
from models.multi_model.multi_model import MultiModel
from evaluation.sampling_helpers import *
from evaluation.plotting import *
from estimate.constants import *


def main():
    cols = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
            "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag', 'redshift']

    Ia_label = 'I, Ia, _ROOT, _SN, _W_UVOPT, Unspecified Ia'
    II_label = 'CC, II, _ROOT, _SN, _W_UVOPT, Unspecified II'

    init_plot_settings()

    model = MultiModel(cols=cols,
                       num_runs=100,
                       class_labels=['Unspecified Ia', 'Unspecified II'],
                       transform_features=True,
                       min_class_size=40,
                       data_file=CUR_DATA_PATH
                       )

    thex_dataset = pd.concat([model.X, model.y], axis=1)
    output_dir = "../figures/testing"

    LSST_results = []
    orig_results = []
    for i in range(model.num_runs):
        # Resample randomly from THEx and LSST data.
        lsst_X, lsst_y, rand_X, rand_y = get_test_sets(thex_dataset=thex_dataset,
                                                       output_dir=output_dir,
                                                       index="",
                                                       num_samples=300)

        # Drop redshift from X's
        if 'redshift' in list(lsst_X):
            print("\n\n Dropping redshift\n")
            lsst_X = lsst_X.drop(labels=['redshift'], axis=1)
            rand_X = rand_X.drop(labels=['redshift'], axis=1)

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

        print("Test counts in rand test set. Ia: " +
              str(Ia_count) + ", II: " + str(II_count))
        print("Test counts in LSST test set. " +
              str(lsst_test_y.groupby(['transient_type']).size()))

        #################
        # Train and test
        model.train_model(train_X.reset_index(drop=True),
                          train_y.reset_index(drop=True))

        LSST_results.append(get_test_performance(lsst_test_X, lsst_test_y, model))
        orig_results.append(get_test_performance(rand_test_X, rand_test_y, model))

    # Save attributes to model for performance measurements
    target_counts = lsst_test_y.groupby(['transient_type']).size()
    II_count = target_counts[II_label]
    Ia_count = target_counts[Ia_label]
    model.class_counts = [Ia_count, II_count]

    # if path.exists('../data/full_model_data.pickle'):
    #     print("Using previous model run data. ")
    #     model = pickle.load(open('../data/full_model_data.pickle', 'rb'))

    plot_performance(model, lsst_test_y, output_dir + "/test_LSST", LSST_results)
    plot_performance(model, rand_test_y, output_dir + "/test_rand", orig_results)
    plot_performance_together(model, lsst_test_y, LSST_results, orig_results)

    # Save data
    with open('../data/full_model_data.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
