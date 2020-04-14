#!/usr/bin/env python

import pickle, glob, sys
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
from astropy.io import ascii

mag_zp = 27.5

if __name__ == '__main__':

    # which band?
    band = sys.argv[1] # str.
    assert band in list('012345')

    lc_prop = OrderedDict()
    p_1st_mjd, p_1st_snr, p_min_snr, p_max_snr, \
            p_1st_mag, p_1st_mag_err, p_min_mag, p_min_mag_err, \
            p_max_mag, p_max_mag_err, p_1st_flux, p_1st_flux_err, \
            p_min_flux, p_min_flux_err, p_max_flux, p_max_flux_err = \
                    list(range(16))

    workdir = '/d2/PLAsTiCC-2018/unblinded/'
    for file_i in glob.glob(workdir + '/plasticc_test_lightcurves_??.csv'):

        print(file_i, '...')

        f = open(file_i, 'r')
        f.readline() # skip header
        # object_id, mjd, passband, flux, flux_err, detected_bool

        for rec_t in f: # for each time point,

            rec_t = rec_t.strip().split(',')
            if rec_t[-1] == '0': continue # not detected.
            if rec_t[ 2] != band: continue # not the right band.

            id_t, mjd_t = int(rec_t[0]), float(rec_t[1])
            flux_t, flux_err_t = float(rec_t[3]), float(rec_t[4])

            snr_t = flux_t / flux_err_t
            mag_t = mag_zp - 2.5 * np.log10(flux_t)
            mag_err_t = 1.08574 / np.abs(snr_t)

            if id_t not in lc_prop: # first epoch.
                lc_prop[id_t] = [mjd_t, snr_t, snr_t, snr_t, \
                        mag_t, mag_err_t, mag_t, mag_err_t, \
                        mag_t, mag_err_t, flux_t, flux_err_t, \
                        flux_t, flux_err_t, flux_t, flux_err_t]
                continue

            # otherwise: get existing results.
            prop_t = lc_prop[id_t]

            # detected at an earlier epoch?
            if mjd_t < prop_t[p_1st_mjd]:
                prop_t[p_1st_snr] = snr_t
                prop_t[p_1st_mag], prop_t[p_1st_mag_err] = mag_t, mag_err_t
                prop_t[p_1st_flux], prop_t[p_1st_flux_err] = flux_t, flux_err_t

            # SNR
            if snr_t > prop_t[p_max_snr]: prop_t[p_max_snr] = snr_t
            if snr_t < prop_t[p_min_snr]: prop_t[p_min_snr] = snr_t

            # flux and mag
            if flux_t > prop_t[p_max_flux]:
                prop_t[p_max_flux], prop_t[p_max_flux_err] = flux_t, flux_err_t
                prop_t[p_max_mag],  prop_t[p_max_mag_err]  = mag_t, mag_err_t
            if flux_t < prop_t[p_min_flux]:
                prop_t[p_min_flux], prop_t[p_min_flux_err] = flux_t, flux_err_t
                prop_t[p_min_mag],  prop_t[p_min_mag_err]  = mag_t, mag_err_t

    with open('lsst-lc-prop-{:}.pk'.format(band), 'wb') as f:
        pickle.dump(lc_prop, f)

if 0:

    cat = np.array(ascii.read(file_i)) # read simulated lc,
    cat = cat[cat['detected_bool'] == 1] # select detected,
    obj_id = cat['object_id']
    snr_acc = np.zeros(obj_id.size, dtype=[('id', 'i4'), ('snr', 'f4')])
    for j_id, p_j in tqdm(enumerate(np.unique(obj_id)), obj_id.size):
        pts_j = cat[obj_id == p_j]
        snr_j = pts_j['flux'] / pts_j['flux_err']
        snr_acc[j_id] = p_j, np.max(snr_j)
    # save snr_acc
    np.save(file_i.replace('.csv', '.snr'), snr_acc)