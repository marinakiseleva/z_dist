
import os
CPU_COUNT = 8
NUM_BINS = 100
TARGET_LABEL = 'transient_type'


THEX_COLOR = "#ffa31a"
LSST_SAMPLE_COLOR = "#24248f"

Z_FEAT = "event_z"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
DATA_DIR = ROOT_DIR + '/../../data/'


# Map from thex names to LSST names
class_mapping = {"Unspecified Ia": "Ia",
                 "II": "II",
                 "SLSN-I": "SLSN-I",
                 "TDE": "TDE",
                 "Ia-91bg": "Ia-91bg",
                 "Ibc": "Ibc"}


CLASS_ID_NAMES = {'Ia': 90, 'II': 42, 'Ibc': 62, 'Ia-91bg': 67, 'TDE': 15, 'SLSN-I': 95}

CUR_DATA_PATH = DATA_DIR + "catalogs/v8/THEx-v8.0-release.mags-xcalib.min-xcal.fits"

# CUR_DATA_PATH = DATA_DIR+"catalogs/v7/THEx-assembled-v7.1a-mags-legacy-xcalib-minxcal.fits"
# DATA_PATH = ROOT_DIR + \
#     "/../../data/catalogs/v7/THEx-assembled-v7.1a-mags-legacy-xcalib-minxcal.fits"
