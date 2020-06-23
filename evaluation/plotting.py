
import numpy as np
import matplotlib.pyplot as plt


def plot_compare_feature_dists(feature_name, class_name, rand_sample, sampled):

    fig, ax = plt.subplots(tight_layout=True, sharex=True,  sharey=True)

    if feature_name == 'redshift':
        bins = np.linspace(0, 1, 20)
        plt.xlim((0, 1))
    else:
        # mag
        bins = np.linspace(9, 23, 50)
        plt.xlim((9, 23))

    b = ax.hist(rand_sample[feature_name].values, bins=bins, density=True,
                label="THEx (random sample)", fill=False, edgecolor='green')
    a = ax.hist(sampled[feature_name].values, bins=bins, density=True,
                label="THEx (LSST sample)", fill=False, edgecolor='red')

    plt.legend()
    plt.title(class_name)
    plt.xlabel(feature_name)
    plt.ylabel("Density")
    plt.savefig("../figures/evaluation/feature_dist_" + feature_name + "_" + class_name)
