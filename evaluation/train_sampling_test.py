"""
Determine if the model does any better or worse with a different distribution of transient events for the training data. Use 2 different training sets and same test set to see if training on LSST-like data results in better performance on LSST-like testing data, than by training on our normal training data and testing on the same LSST-like data. IE - the most 'LSST-like' data.
"""

# Sample training set
from models.multi_model.multi_model import MultiModel
from evaluation.sampling_test import get_test_performance, plot_performance
import pandas as pd
from evaluation.sampling_test import get_THEx_sampled_data, get_test_sets


def main():
    cols = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
            "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag', 'redshift']
    model = MultiModel(cols=cols,
                       class_labels=['Unspecified Ia', 'Unspecified II'],
                       transform_features=False,
                       min_class_size=40
                       )

    thex_dataset = pd.concat([model.X, model.y], axis=1)
    output_dir = "../figures/testing"
    lsst_X, lsst_y, rand_X, rand_y = get_test_sets(thex_dataset=thex_dataset,
                                                   output_dir=output_dir,
                                                   index="",
                                                   num_samples=200)

    model.num_runs = 10
    model.num_folds = None
    LSST_results = []
    orig_results = []
    for i in range(model.num_runs):

        # Train on 80% LSST, and test on remaining 20%
        # sample from rand X and y to get equal class counts as in test_LSST
        # compare 2 test set results
        train_test_size = int(lsst_X.shape[0] * 0.8)
        train_X = lsst_X.sample(train_test_size)
        train_y = lsst_y.iloc[train_X.index].reset_index(drop=True)
        train_X = train_X.reset_index(drop=True)

        # LSST testing is remaining LSST data
        LSST_test_X = lsst_X.loc[~lsst_X.index.isin(train_X.index)]
        LSST_test_y = lsst_y.iloc[LSST_test_X.index].reset_index(drop=True)
        LSST_test_X = LSST_test_X.reset_index(drop=True)

        # Get class counts from LSST test set
        Ia_label = 'I, Ia, _ROOT, _SN, _W_UVOPT, Unspecified Ia'
        II_label = 'CC, II, _ROOT, _SN, _W_UVOPT, Unspecified II'
        counts = LSST_test_y.groupby(['transient_type']).size()
        II_count = counts[II_label]
        Ia_count = counts[Ia_label]

        # Sample from THEx with same counts
        Ia_rand_y = rand_y.loc[rand_y['transient_type'] == Ia_label].sample(Ia_count)
        Ia_rand_X = rand_X.iloc[Ia_rand_y.index]
        II_rand_y = rand_y.loc[rand_y['transient_type'] == II_label].sample(II_count)
        II_rand_X = rand_X.iloc[II_rand_y.index]
        rand_test_y = pd.concat([Ia_rand_y, II_rand_y]).reset_index(drop=True)
        rand_test_X = pd.concat([Ia_rand_X, II_rand_X]).reset_index(drop=True)

        model.train_model(train_X, train_y)

        LSST_results.append(get_test_performance(LSST_test_X, LSST_test_y, model))
        orig_results.append(get_test_performance(rand_test_X, rand_test_y, model))

    plot_performance(model, LSST_test_y, "../figures/testing/test_LSST", LSST_results)
    plot_performance(model, rand_test_y, "../figures/testing/test_rand", orig_results)


if __name__ == "__main__":
    main()
