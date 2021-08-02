"""
See how testing sets and training set used in evaluation differ in low-dimensional reps.
"""
import os
import pickle
from os import path
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from thex_data.data_consts import *
import matplotlib.pyplot as plt
from pylab import rcParams


from models.multi_model.multi_model import MultiModel
from evaluation.sampling_test import get_test_performance, get_test_sets
from evaluation.plotting import *
from estimate.constants import *


FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 600

LSST_TEST_STR = "LSST-like test set"
THEX_TEST_STR = "THEx test set"
TRAIN_SET_STR = "THEx training set"


Ia_label = 'I, Ia, _ROOT, _SN, _W_UVOPT, Unspecified Ia'
II_label = 'CC, II, _ROOT, _SN, _W_UVOPT, Unspecified II'

p_colors = {LSST_TEST_STR: "#24248f",
            THEX_TEST_STR: "#ffa31a",
            TRAIN_SET_STR: "black"}


def plot_reduction(ax, data, num_features, data_type, output_dir):
    """
    Distinguish classes in reduced space with different colors.
    """

    rcParams['figure.figsize'] = 6, 6
    Ia_data = data[data['transient_type'] == Ia_label]
    II_data = data[data['transient_type'] == II_label]

    ax.scatter(Ia_data['x'], Ia_data['y'], color="#54534f",
               label="Ia (unspec.)", marker="+")
    ax.scatter(II_data['x'], II_data['y'], color="#b5b3aa",
               label="II (unspec.)", marker="x")

    for k in ax.spines.keys():
        ax.spines[k].set_color(p_colors[data_type])
        ax.spines[k].set_linewidth(2)

    ax.set_xlabel('x reduction', fontsize=16)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)


def fit_and_plot(axis, X, y, data_type, output_dir, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=10000, n_iter_without_progress=300):

    print("\n\nEvaluating " + str(data_type) + "\n\n")

    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                learning_rate=learning_rate,
                n_iter=n_iter,
                n_iter_without_progress=n_iter_without_progress,
                random_state=10,
                verbose=2,
                n_jobs=4)

    embedding = tsne.fit_transform(X)

    reduced_data = pd.DataFrame(embedding, columns=['x', 'y'])
    fulldata = pd.concat([reduced_data, y], axis=1)
    plot_reduction(axis, fulldata, len(list(X)), data_type, output_dir)

    return tsne


def main():
    cols = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
            "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag', Z_FEAT]

    if path.exists('../data/modelsave.pickle'):
        with open('../data/modelsave.pickle', 'rb') as handle:
            model = pickle.load(handle)
    else:
        model = MultiModel(cols=cols,
                           num_runs=2,
                           class_labels=['Unspecified Ia', 'Unspecified II'],
                           transform_features=True,
                           min_class_size=40,
                           data_file=CUR_DATA_PATH
                           )
        # Save data
        with open('../data/modelsave.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    thex_dataset = pd.concat([model.X, model.y], axis=1)
    output_dir = "../figures/testing"
    lsst_X, lsst_y, rand_X, rand_y = get_test_sets(thex_dataset=thex_dataset,
                                                   output_dir=output_dir,
                                                   index="",
                                                   num_samples=300)

    # Drop redshift from X's
    if Z_FEAT in list(lsst_X):
        print("\n\n Dropping redshift\n")
        lsst_X = lsst_X.drop(labels=[Z_FEAT], axis=1)
        rand_X = rand_X.drop(labels=[Z_FEAT], axis=1)

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

    n_iter = 50000
    n_iter_without_progress = 1000
    early_exaggeration = 12
    perplexity = 5
    learning_rate = 10

    np.set_printoptions(precision=3)

    # fit_and_plot(X=train_X,
    #              y=train_y,
    #              data_type=TRAIN_SET_STR,
    #              output_dir=output_dir,
    #              perplexity=perplexity,
    #              early_exaggeration=early_exaggeration,
    #              learning_rate=learning_rate,
    #              n_iter=n_iter,
    #              n_iter_without_progress=n_iter_without_progress)
    # fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
    #                        dpi=100, tight_layout=True)

    fig, ax = plt.subplots(figsize=(8, 3),
                           nrows=1, ncols=2,
                           dpi=200)

    fit_and_plot(axis=ax[0],
                 X=rand_test_X,
                 y=rand_test_y,
                 data_type=THEX_TEST_STR,
                 output_dir=output_dir,
                 perplexity=perplexity,
                 early_exaggeration=early_exaggeration,
                 learning_rate=learning_rate,
                 n_iter=n_iter,
                 n_iter_without_progress=n_iter_without_progress)

    fit_and_plot(axis=ax[1],
                 X=lsst_test_X,
                 y=lsst_test_y,
                 data_type=LSST_TEST_STR,
                 output_dir=output_dir,
                 perplexity=perplexity,
                 early_exaggeration=early_exaggeration,
                 learning_rate=learning_rate,
                 n_iter=n_iter,
                 n_iter_without_progress=n_iter_without_progress)

    ax[1].legend(fontsize=18, bbox_to_anchor=(1.1, 0.7),
                 labelspacing=.2, handlelength=1)

    ax[0].set_ylabel('y reduction', fontsize=16)
    ax[0].set_title(THEX_TEST_STR, fontsize=18, color=THEX_COLOR)
    ax[1].set_title("LSST-like test set", fontsize=18,  color=LSST_SAMPLE_COLOR)
    plt.savefig(output_dir + "/tsne_red.pdf", bbox_inches='tight')

    # with open('../data/modelsave.pickle', 'wb') as handle:
    #         pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
