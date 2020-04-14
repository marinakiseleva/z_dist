#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from collections import OrderedDict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from astropy.cosmology import WMAP9 as cosmo
from scipy.ndimage.filters import gaussian_filter1d

from matplotlib.font_manager import FontProperties

DATA_DIR = "data/"


def make_kde(val, rg=None, bins=None,):
    ax = np.linspace(rg[0], rg[1], 2 * bins + 1)[1::2]
    kde = gaussian_kde(val[np.isfinite(val)],)
    return ax, kde(ax)


def get_attributes(data_file, rowmask_file):
    """
    Uses redshift and labels & returns 
    redshifts, valid redshifts, and idxf
    idxf is  map from each class name to  a list that is the same length as the number of samples in the dataset. for each one, it is True or False, based on whether that sample contains that label.
    """
    data = np.load(DATA_DIR + data_file, allow_pickle=True)
    rowmask = np.load(DATA_DIR + rowmask_file, allow_pickle=True)
    z = data['redshift']
    valid_z = np.isfinite(z)
    idxf = {r: rowmask[r] == 1 for r in list(rowmask.dtype.names)}
    return z, valid_z, idxf


if __name__ == '__main__':
    """
    Catalog is assembled-v6.npy which is a Numpy recarray, it contains a column for 'redshift' which has the corresponding redshift for each sample

    'typerowmask-v6.npy' contains information about the labels for each sample in the catalog. It's basically a list of one hot vectors. The ordering of each Numpy array is the same as rowmask.dtype.names and idxf.keys()

    """

    zred, valid_z, idxf = get_attributes(
        data_file='orig_assembled-v6.npy',
        rowmask_file='orig_typerowmask-v6.npy')

    allmags_zred, allmags_valid_z, allmags_idxf = get_attributes(
        data_file='allmags_data.npy',
        rowmask_file='allmags_rowmask.npy')

    submags_zred, submags_valid_z, submags_idxf = get_attributes(
        data_file='subsetmags_data.npy',
        rowmask_file='subsetmags_rowmask.npy')

    # read simulated LSST events.
    with open('lsst-sims.pk', 'rb') as f:
        lsst_sims = pickle.load(f)

    bin_range = {
        'Ia':       (0., 1.25),
        'Ia-91bg':  (0., 0.60),
        'Ia-02cx':  (0., 0.80),
        'II':       (0., 1.00),
        'Ibc':      (0., 0.80),
        'SLSN-I':   (0., 3.25),
        'TDE':      (0., 1.75),
    }
    N_bins = {
        'Ia': 48,
        'Ia-91bg': 32,
        'Ia-02cx': 24,
        'II':  48,
        'Ibc': 32,
        'SLSN-I': 16,
        'TDE': 24,
    }
    alt_keys = {
        'Ibc': idxf['SE'],
        'TDE': idxf['TDE'],
    }

    allmags_alt_keys = {
        'Ibc': allmags_idxf['SE'],
        'TDE': allmags_idxf['TDE'],
    }
    submags_alt_keys = {
        'Ibc': submags_idxf['SE'],
        'TDE': submags_idxf['TDE'],
    }

    # draw panels.
    # fig = plt.figure(figsize=(12., 12.), dpi=80, sharex=True)
    fig, axes = plt.subplots(3, 2,
                             sharex=True,
                             sharey=True,
                             figsize=(12, 12))

    cmap = mpl.cm.get_cmap('Set1')

    classes = ['Ia', 'Ia-91bg', 'II', 'Ibc', 'SLSN-I', 'TDE']
    i_tp = 0
    for i, row in enumerate(axes):
        for j, cell in enumerate(row):
            tp_i = classes[i_tp]
            i_tp += 1
            objid_i, zred_tp_i, snr_max_i, snr_min_i = lsst_sims[tp_i]
            idxf_i = alt_keys[tp_i] if tp_i in alt_keys else idxf[tp_i]

            allmags_idxf_i = allmags_alt_keys[
                tp_i] if tp_i in allmags_alt_keys else allmags_idxf[tp_i]

            submags_idxf_i = submags_alt_keys[
                tp_i] if tp_i in submags_alt_keys else submags_idxf[tp_i]

            axi = axes[i, j]
            if i == 2:
                axi.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
                axi.set_xlabel("Redshift")
            if j == 0:
                axi.set_ylabel("Normalized density")

            fixed_width = 1.2

            # All mags
            axi.hist(allmags_zred[allmags_valid_z * allmags_idxf_i],
                     color="#4187e8",
                     label=tp_i + ' (all-features)',
                     range=bin_range[tp_i],
                     bins=N_bins[tp_i],
                     density=True,
                     histtype='bar',
                     stacked=True)
            axi.hist(submags_zred[submags_valid_z * submags_idxf_i],
                     color="#66ff66",
                     label=tp_i + ' (g-W2)',
                     range=bin_range[tp_i],
                     bins=N_bins[tp_i],
                     density=True,
                     histtype='bar',
                     stacked=True)

            axi.hist(zred_tp_i,
                     color="#ff99ff",
                     label=tp_i + ' (LSST, All)',
                     range=bin_range[tp_i],
                     bins=N_bins[tp_i], density=True,
                     histtype='step', stacked=True)

            axi.hist(zred_tp_i[snr_min_i > 25],
                     color="#660066",
                     linewidth=fixed_width,
                     label=tp_i + ' (LSST, SNR > 25)',
                     range=bin_range[tp_i],
                     bins=N_bins[tp_i], density=True,
                     histtype='step', stacked=True)
            axi.legend(loc='upper right')

    plt.xlim(0, 1.5)
    plt.xticks(np.linspace(0, 1.5, 15))

    plt.savefig('redshift_dist_data_vs_lsst.png')
    plt.show()
