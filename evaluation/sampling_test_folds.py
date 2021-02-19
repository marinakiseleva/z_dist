"""
Determine if the model does any better or worse with a different distribution of transient events, for the testing data. 

Use unique test sets each time. 
"""
import warnings
warnings.filterwarnings("ignore")


from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
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
    Return X and y of LSST and random sampled testing sets.
    """
    Ia_LSST, Ia_THEx, Ia_LSST_Z, test_set = sample_data(class_name="Ia",
                                                        num_samples=num_samples,
                                                        thex_dataset=thex_dataset,
                                                        i=index)
    # if Ia_LSST.shape[0] != num_samples:
    #     print("Resetting sample count to match Ia.")
    #     num_samples = Ia_LSST.shape[0]

    II_LSST, II_THEx, II_LSST_Z, test_set = sample_data(class_name="II",
                                                        num_samples=num_samples,
                                                        thex_dataset=thex_dataset,
                                                        i=index)

    print("\nSampled " + str(Ia_LSST.shape[0]) + " Ia, " + str(II_LSST.shape[0]) + " II")

    plot_sample_dists_together(Ia_LSST,
                               Ia_THEx,
                               Ia_LSST_Z,
                               II_LSST,
                               II_THEx,
                               II_LSST_Z,
                               output_dir,
                               index=index)

    LSST_X, LSST_y = get_source_target(
        pd.concat([Ia_LSST, II_LSST]))
    THEx_X, THEx_y = get_source_target(
        pd.concat([Ia_THEx, II_THEx]))
    return LSST_X, LSST_y, THEx_X, THEx_y, test_set


def plot_sample_dist(ax, rand_sample, lsst_sample, lsst_orig, class_name):

    THEX_COLOR = "#ffa31a"
    LSST_COLOR = "#80ccff"
    LSST_SAMPLE_COLOR = "#24248f"

    def plot_step(data, bins, axis, label, color):
        """
        Get hist data and plot as step graph, no inner lines
        """
        # bins values in bin are (for left, right), left <= x < right
        vals, bins = np.histogram(data, bins=bins, density=True)
        a = np.array([0])
        bin_indices = np.linspace(min(bins), max(bins), len(bins))
        bin_indices = bin_indices[1:]
        xnew = np.concatenate((a, bin_indices), axis=0)
        ynew = np.concatenate((a, vals), axis=0)

        axis.step(x=xnew,
                  y=ynew,
                  label=label,
                  linewidth=2,
                  color=color)

    Z_bins = np.linspace(0, 1, 50)

    a = ax.hist(lsst_orig,
                density=True,
                bins=Z_bins,
                label="LSST",
                fill=True,
                alpha=0.8,
                color=LSST_COLOR)

    plot_step(data=rand_sample['redshift'].values,
              bins=Z_bins,
              axis=ax,
              label="THEx test set",
              color=THEX_COLOR)
    plot_step(data=lsst_sample['redshift'].values,
              bins=Z_bins,
              axis=ax,
              label="LSST-like test set",
              color=LSST_SAMPLE_COLOR)

    ax.set_title(class_name, fontsize=22, y=0.8, x=0.75)


def plot_sample_dists_together(Ia_sampled, Ia_rand_sample, Ia_LSST_Z, II_sampled, II_rand_sample, II_LSST_Z, output_dir, index):
    """
    Plot LSST orig vs THEx sample vs LSST sample for each class on shared fig.
    """
    # Plot LSST data, sampled LSST, and random sample
    rc('font', family="Times New Roman")
    f, ax = plt.subplots(nrows=2,
                         ncols=1,
                         sharex=True, sharey=True,
                         figsize=(5, 7),
                         dpi=140)

    plot_sample_dist(ax[0], Ia_rand_sample, Ia_sampled, Ia_LSST_Z, "Ia (unspec.)")
    plot_sample_dist(ax[1], II_rand_sample, II_sampled, II_LSST_Z, "II (unspec.)")

    ax[0].legend(fontsize=14, loc="upper left",  labelspacing=.2, handlelength=1)

    ax[0].yaxis.set_tick_params(labelsize=16)
    ax[1].yaxis.set_tick_params(labelsize=16)
    ax[1].xaxis.set_tick_params(labelsize=16)

    ax[1].set_xlabel("Redshift", fontsize=20)
    ax[0].set_ylabel("Density", fontsize=20)
    ax[1].set_ylabel("Density", fontsize=20)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_dir + "/samples_" + str(index) + ".pdf", bbox_inches='tight')


def sample_data(class_name, num_samples, thex_dataset,  i=""):
    """
    Create 2 sample test sets from THEx data, one randomly sampled from our data and the other sampled with LSST redshift dist. Return test sets AND the dataset we sampled from, with those test samples now removed. 
    :param class_name: Name of class we are sampling. Only handles Ia and II.
    :param thex_dataset: DataFrame of THEx data, X and y
    :param i: iteration 
    """

    thex_class_data = get_thex_class_data("Unspecified " + class_name, thex_dataset)

    # Pull down LSST data
    feature_name = "r_first_mag"
    lsst_class_data = get_lsst_class_data(class_name, feature_name)

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
            print("Not enough in this range by " + str(int(f_df.shape[0] - samples)))
            lsst_sample.append(f_df)

    lsst_sample = pd.concat(lsst_sample)
    class_count = lsst_sample.shape[0]
    random_sample = thex_class_data.sample(class_count)

    # # Dropping the test set samples from the whole training set.
    # LSST_indices = list(lsst_sample.index.values)
    # rand_indices = list(random_sample.index.values)
    # all_indices = list(set(LSST_indices + rand_indices))
    # orig_size = thex_dataset.shape[0]
    # thex_dataset.drop(index=all_indices, inplace=True)
    # new_size = thex_dataset.shape[0]
    # print("Whole test set goes from size " + str(orig_size) + " to size " + str(new_size))

    lsst_sample.reset_index(drop=True, inplace=True)
    random_sample.reset_index(drop=True, inplace=True)

    return lsst_sample, random_sample, lsst_z_vals, thex_dataset


def get_test_results(model, output_dir, iterations=100):
    """
    Train on model data and test on passed in data for X trials, and visualize results.
    """
    model.num_runs = iterations
    model.num_folds = None
    thex_dataset = pd.concat([model.X, model.y], axis=1)

    LSST_results = []
    orig_results = []

    whole_test_set = thex_dataset.copy(True)
    for i in range(model.num_runs):
        # Resample testing sets each run
        print("\n\nIteration " + str(i))
        print("whole test set size " + str(whole_test_set.shape[0]))
        X_lsst, y_lsst, X_orig, y_orig, whole_test_set = get_test_sets(
            whole_test_set,
            output_dir,
            i,
            num_samples=100)

        # Update training data to remove testing sets
        X_train, y_train = get_training_data(X_lsst, X_orig, model.X, model.y)

        # Ensure all X sets have columns in same order (and no redshift as feature)
        X_lsst = X_lsst[ordered_mags]
        X_orig = X_orig[ordered_mags]
        X_train = X_train[ordered_mags]

        print("\nTraining set size: " + str(X_train.shape[0]))
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

    plot_performance_together(model, y_lsst, LSST_results, orig_results)


def main():

    # Initialize output directory

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
    output_dir = ROOT_DIR + "/figures/testing/" + dt_string
    os.mkdir(output_dir)

    init_plot_settings()

    cols = ["g_mag", "r_mag", "i_mag", "z_mag", "y_mag",
            "W1_mag", "W2_mag", "H_mag", "K_mag", 'J_mag',
            'redshift']
    model = MultiModel(cols=cols,
                       class_labels=['Unspecified Ia', 'Unspecified II'],
                       transform_features=False,
                       min_class_size=40,
                       data_file=CUR_DATA_PATH
                       )
    model.dir = output_dir

    get_test_results(model=model,
                     output_dir=output_dir,
                     iterations=10)


if __name__ == "__main__":
    main()
