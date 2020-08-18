from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams

import os
import random
import math

from models.multi_model.multi_model import MultiModel
from evaluation.plotting import *
from estimate.get_data import *
from evaluation.sampling_test import *

FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 600

LSST_TEST_STR = "Rubin-like test set"
THEX_TEST_STR = "THEx test set"
TRAIN_SET_STR = "THEx training set"

p_colors = {LSST_TEST_STR: "#ffb3b3",
            THEX_TEST_STR: "#b3e6b3",
            TRAIN_SET_STR: "black"}


def get_params(tsne):
    return tsne.get_params(deep=True)


def plot_tsne(embedding, dimensions, num_features):
    rcParams['figure.figsize'] = 6, 6
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.xlabel('x reduction')
    plt.ylabel('y reduction')
    plot_title = "t-SNE Embedding of " + \
        str(num_features) + " Features in " + str(dimensions) + \
        " Dimensions"
    plt.title(plot_title)
    plt.show()


def run_tsne(data, dimensions=2, perplexity=5, early_exaggeration=12.0, learning_rate=60, n_iter=3000, n_iter_without_progress=400, random_state=10):
    """
    Runs t-SNE on data, reduce to # of dimensions passed in.
    :param data: DF of complete data, < 50 features
    :param dimensions: Number of dimensions to reduce to
    :param perplexity: Number of nearest neighbors, between 5 and 50
    :param early_exaggeration: How tight clusters
    """
    num_features = len(list(data))
    tsne = TSNE(n_components=dimensions,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                learning_rate=learning_rate,
                n_iter=n_iter,
                n_iter_without_progress=n_iter_without_progress,
                random_state=random_state)
    embedding = tsne.fit_transform(data)
    plot_tsne(embedding, dimensions, num_features)
    return embedding

from thex_data.data_consts import *


def plot_reduction(data, num_features, data_type, output_dir):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT),
                           dpi=DPI, tight_layout=True)

    rcParams['figure.figsize'] = 6, 6
    Ia_data = data[data['transient_type'] ==
                   'I, Ia, _ROOT, _SN, _W_UVOPT, Unspecified Ia']

    II_data = data[data['transient_type'] ==
                   'CC, II, _ROOT, _SN, _W_UVOPT, Unspecified II']

    ax.scatter(Ia_data['x'], Ia_data['y'], color="red", label="Ia")
    ax.scatter(II_data['x'], II_data['y'], color="blue", label="II")

    for k in ax.spines.keys():
        ax.spines[k].set_color(p_colors[data_type])

    plt.xlabel('x reduction', fontsize=LAB_S)
    plt.ylabel('y reduction', fontsize=LAB_S)

    plt.title(data_type, fontsize=TITLE_S)
    plt.legend(fontsize=LAB_S)
    plt.savefig(output_dir + "/" + data_type)


def fit_and_plot(X, y, data_type, output_dir, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=10000, n_iter_without_progress=300):

    print("\n\nEvaluating " + str(data_type) + "\n\n")

    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                early_exaggeration=early_exaggeration,
                learning_rate=learning_rate,
                n_iter=n_iter,
                n_iter_without_progress=n_iter_without_progress,
                random_state=10,
                verbose=1,
                n_jobs=4)

    embedding = tsne.fit_transform(X)

    reduced_data = pd.DataFrame(embedding, columns=['x', 'y'])
    fulldata = pd.concat([reduced_data, y], axis=1)
    plot_reduction(fulldata, len(list(X)), data_type, output_dir)

    return tsne

from datetime import datetime


def main():
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

    lsst_sampled_X = lsst_sampled_X[ordered_mags]
    orig_sampled_X = orig_sampled_X[ordered_mags]

    # perplexity is 5- 50, ~number of neighbors per pixel

    #  learning_rate  [10.0, 1000.0]
    n_iter = 50000
    n_iter_without_progress = 400

    np.set_printoptions(precision=3)

    fit_and_plot(X=model.X,
                 y=model.y,
                 data_type=TRAIN_SET_STR,
                 output_dir=output_dir,
                 perplexity=80,
                 early_exaggeration=10.0,
                 learning_rate=20,
                 n_iter=n_iter,
                 n_iter_without_progress=n_iter_without_progress)

    fit_and_plot(X=orig_sampled_X,
                 y=orig_sampled_y,
                 data_type=THEX_TEST_STR,
                 output_dir=output_dir,
                 perplexity=15,
                 early_exaggeration=2,
                 learning_rate=10,
                 n_iter=n_iter,
                 n_iter_without_progress=n_iter_without_progress)

    fit_and_plot(X=lsst_sampled_X,
                 y=lsst_sampled_y,
                 data_type=LSST_TEST_STR,
                 output_dir=output_dir,
                 perplexity=10,
                 early_exaggeration=5,
                 learning_rate=10,
                 n_iter=n_iter,
                 n_iter_without_progress=n_iter_without_progress)


if __name__ == "__main__":
    main()
