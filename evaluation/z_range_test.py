import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import random
import math

from models.multi_model.multi_model import MultiModel
from evaluation.plotting import *
from estimate.get_data import *
from evaluation.sampling_test import *


# See if different redshift ranges of data perform differently.
# Train on X % of data; of remaining y% of data, split by redshift range.
# Then, plot test performance of each redshift range separately (0-0.2,
# 0.2-0.4, .etc.)


exp = str(random.randint(1, 10**10))
output_dir = "../figures/evaluation/" + exp
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

class_count = 600

Ia_sampled, Ia_rand_sample = get_THEx_sampled_data(class_name="Ia",
                                                   max_rmag=None,
                                                   num_samples=class_count,
                                                   thex_dataset=thex_dataset,
                                                   output_dir=output_dir)
II_sampled, II_rand_sample = get_THEx_sampled_data(class_name="II",
                                                   max_rmag=None,
                                                   num_samples=class_count,
                                                   thex_dataset=thex_dataset,
                                                   output_dir=output_dir)
orig_sampled_X, orig_sampled_y = get_source_target(
    pd.concat([Ia_rand_sample, II_rand_sample]))


# Update training data to remove testing set
X_train = model.X.copy()
y_train = model.y.copy()
print("Original size of training set " + str(X_train.shape[0]))

train_X, train_y = remove_data(alt_X=orig_sampled_X,
                               orig_X=X_train,
                               orig_y=y_train)

print("New size of training set " + str(train_X.shape[0]))


df = pd.concat([model.X, model.y], axis=1)
from thex_data.data_plot import plot_feature_distribution
plot_feature_distribution(model_dir=model.dir,
                          df=df,
                          feature='redshift',
                          class_labels=model.class_labels)


# Reorder all X to have same order
train_X = train_X[ordered_mags]

model.train_model(train_X, train_y)


# Split test set into 3 redshift ranges

co1 = 0.04  # Cut off 1

co2 = 0.2  # Cut off 2

low_r_X = orig_sampled_X.loc[orig_sampled_X['redshift'] < co1]
low_r_y = orig_sampled_y.loc[orig_sampled_X['redshift'] < co1]

med_r_X = orig_sampled_X.loc[(orig_sampled_X['redshift'] >= co1)
                             & (orig_sampled_X['redshift'] < co2)]
med_r_y = orig_sampled_y.loc[(orig_sampled_X['redshift'] >= co1)
                             & (orig_sampled_X['redshift'] < co2)]


high_r_X = orig_sampled_X.loc[orig_sampled_X['redshift'] >= co2]
high_r_y = orig_sampled_y.loc[orig_sampled_X['redshift'] >= co2]

print("Sizes")
print("Low " + str(low_r_X.shape[0]))

print("Mid " + str(med_r_X.shape[0]))

print("High " + str(high_r_X.shape[0]))


FIG_WIDTH = 6
FIG_HEIGHT = 4
DPI = 600
fig, ax = plt.subplots(tight_layout=True, sharex=True,  sharey=True,
                       figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

bins = np.linspace(0, 1, 20)
plt.xlim((0, 1))

low = ax.hist(low_r_X['redshift'].values, bins=bins,
              label="Low z", fill=False, edgecolor='green')
mid = ax.hist(med_r_X['redshift'].values, bins=bins,
              label="Mid z", fill=False, edgecolor='purple')
high = ax.hist(high_r_X['redshift'].values, bins=bins,
               label="Mid z", fill=False, edgecolor='blue')

plt.legend()
plt.title('Redshift ranges of test set', fontsize=14)
plt.xlabel('redshift', fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.savefig(output_dir + "/z_ranges")


def evaluate(X_test, y_test, test_type, model):
    results = []
    probabilities = model.get_all_class_probabilities(X_test, model.normalize)
    # Add labels as column to probabilities, for later evaluation
    probabilities = np.hstack((probabilities, y_test))
    results.append(probabilities)
    orig_model_dir = model.dir
    model.dir = orig_model_dir + "/" + str(test_type)
    os.mkdir(model.dir)
    model.results = results
    model.y = y_test
    model.visualize_performance()
    model.dir = orig_model_dir

X = high_r_X
y = high_r_y
ttype = "high"

evaluate(X_test=X[ordered_mags],
         y_test=y['transient_type'].values.reshape(-1, 1),
         test_type=ttype,
         model=model)

X = med_r_X
y = med_r_y
ttype = "mid"

evaluate(X_test=X[ordered_mags],
         y_test=y['transient_type'].values.reshape(-1, 1),
         test_type=ttype,
         model=model)

X = low_r_X
y = low_r_y
ttype = "low"

evaluate(X_test=X[ordered_mags],
         y_test=y['transient_type'].values.reshape(-1, 1),
         test_type=ttype,
         model=model)
