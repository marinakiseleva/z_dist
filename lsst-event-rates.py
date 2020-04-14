#!/usr/bin/env python

import pickle
from collections import OrderedDict

import numpy as np
from astropy.io import ascii

if __name__ == '__main__':

    # meta data file.
    meta_file = '/d2/PLAsTiCC-2018/unblinded/plasticc_test_metadata.csv'
    cat = np.array(ascii.read(meta_file))

    # only events in shallow survey
    cat = cat[cat['ddf_bool'] == 0]

    # names and numeric labels
    event_types = [('Ia', 90), ('Ia-91bg', 67), ('Ia-02cx', 52),
            ('II', 42), ('Ibc', 62), ('SLSN-I', 95), ('TDE', 15), ('KN', 64)]

    # read SNR, flux and magnitude
    lc_prop = OrderedDict()
    for i_band, band_i in enumerate('ugrizy'):
        with open('lsst-lc-prop-{:}.pk'.format(i_band), 'rb') as f:
            lc_prop[band_i] = pickle.load(f)

    fields = [
        'first_mjd', 'first_snr', 'min_snr', 'max_snr',
        'first_mag', 'first_mag_err', 'min_mag', 'min_mag_err',
        'max_mag', 'max_mag_err', 'first_flux', 'first_flux_err',
        'min_flux', 'min_flux_err', 'max_flux', 'max_flux_err'
    ]
    nan_array = [np.nan for _ in fields]
    # in case any band is not available.

    lsst_type_prop = OrderedDict()
    for event_type_i, label_i in event_types:

        print('Type:', event_type_i)

        # select events of this type.
        type_filter_i = cat['true_target'] == label_i
        obj_id_i = cat['object_id'][type_filter_i]

        # properties of this event type.
        type_prop_i = dict(
            obj_id=obj_id_i,
            true_z=cat['true_z'][type_filter_i],
            photo_z=cat['hostgal_photoz'][type_filter_i]
        )

        # SNR, flux and mag in each band,
        for band_j, lc_prop_j in lc_prop.items():
            print('  Band:', band_j)
            prop_sub_j = [lc_prop_j[w] if w in lc_prop_j \
                          else nan_array \
                          for w in obj_id_i]
            for k_field, field_k in enumerate(fields):
                print('    Column:', field_k)
                arr_k = np.array([w[k_field] for w in prop_sub_j])
                type_prop_i[band_j + '_' + field_k] = arr_k

        lsst_type_prop[event_type_i] = type_prop_i

    # save.
    with open('lsst-sims.pk', 'wb') as f:
        pickle.dump(lsst_type_prop, f)