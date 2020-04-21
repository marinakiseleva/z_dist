#!/usr/bin/env python

import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # read light curve properties.

    with open('data/lsst-sims.pk', 'rb') as f:
        lc_prop = pickle.load(f)
    # this is a nested dictionary.

    # -------------------------------------------------------------------------
    # list available types.
    # print(list(lc_prop.keys()))

    # keys are:
    # 'Ia', 'Ia-91bg', 'Ia-02cx', 'II', 'Ibc', 'SLSN-I', 'TDE', 'KN'

    # -------------------------------------------------------------------------
    # For any type, list keys for light curve properties
    # lc_prop_Ibc = lc_prop['Ia']
    # print(list(lc_prop_Ibc))

    # keys are:
    # obj_id:                         light curve id
    # true_z, photo_z:                transient redshift and host photo-z

    # These columns are calculated for each band (* = u, g, r, i, z, y)

    # *_first_mjd:                    epoch of initial detection ('first epoch')
    # *_first_snr:                    first-epoch SNR
    # *_min_snr, *_max_snr:           minimal and maximal SNR of the light curve
    # *_first_mag, *_first_mag_err:   first-epoch magnitude and error
    # *_min_mag, *_min_mag_err:       faintest magnitude and error
    # *_max_mag, *_max_mag_err:       peak or brighest magnitude and error
    # *_first_flux, *_first_flux_err: first-epoch physical flux and error
    # *_min_flux, *_min_flux_err:     minimal flux (matching faintest magnitude)
    # *_max_flux, *_max_flux_err:     maximal flux (matching peak magnitude)

    # -------------------------------------------------------------------------
    # Example: distribution of first-epoch r-band magnitude for each type.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for type_i, lc_prop_i in lc_prop.items():
        ax.hist(lc_prop_i['r_first_mag'], range=(18., 26.5),
                bins=64, histtype='step', normed=True,
                label=type_i,)
    plt.title('First-epoch r-band magnitude')
    plt.legend()
    plt.savefig('1.pdf')
    plt.show()

    # -------------------------------------------------------------------------
    # Example: For SNe Ia, first-epoch magnitude in each band.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for band_i in 'ugrizy':
        ax.hist(lc_prop['Ia'][band_i + '_first_mag'], range=(18., 26.5),
                bins=64, histtype='step', normed=True,
                label=band_i,)
    plt.title('First-epoch magnitude in each band (SNe Ia)')
    plt.legend()
    plt.savefig('2.pdf')
    plt.show()

    # -------------------------------------------------------------------------
    # Example: redshifts of Ia with first-epoch magnitude brigher than a limit.

    lc_prop_Ia = lc_prop['Ia']

    rmag_24 = lc_prop_Ia['r_first_mag'] < 24.0
    rmag_22 = lc_prop_Ia['r_first_mag'] < 22.0
    rmag_20 = lc_prop_Ia['r_first_mag'] < 20.0

    zred_24 = lc_prop_Ia['true_z'][rmag_24]
    zred_22 = lc_prop_Ia['true_z'][rmag_22]
    zred_20 = lc_prop_Ia['true_z'][rmag_20]

    # plot distribution.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(zred_24, range=(0., 1.), bins=64,
            histtype='step', normed=True, label='r_init < 24')
    ax.hist(zred_22, range=(0., 1.), bins=64,
            histtype='step', normed=True, label='r_init < 22')
    ax.hist(zred_20, range=(0., 1.), bins=64,
            histtype='step', normed=True, label='r_init < 20')
    plt.title('First-epoch magnitude cut & SN Ia redshift distribution')
    plt.legend()
    plt.savefig('3.pdf')
    plt.show()

    # -------------------------------------------------------------------------
    # Example: For SNe Ia, first-epoch magnitude vs. redshift.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist2d(lc_prop['Ia']['r_first_mag'], lc_prop['Ia']['true_z'],
              range=((18., 26.5), (0., 1.25)), bins=64, cmap='Blues')
    plt.title('First-epoch r-band magnitude vs. redshift (SNe Ia)')
    plt.xlabel('r_first'), plt.ylabel('redshift')
    plt.savefig('4.pdf')
    plt.show()
