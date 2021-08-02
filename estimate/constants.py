
import os
CPU_COUNT = 8
NUM_BINS = 100
TARGET_LABEL = 'transient_type'


THEX_COLOR = "#ffa31a"
LSST_SAMPLE_COLOR = "#24248f"

Z_FEAT = "event_z"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
DATA_DIR = ROOT_DIR + '/../../data/'


CUR_DATA_PATH = DATA_DIR+ "catalogs/v8/THEx-v8.0-release.mags-xcalib.min-xcal.fits"

# CUR_DATA_PATH = DATA_DIR+"catalogs/v7/THEx-assembled-v7.1a-mags-legacy-xcalib-minxcal.fits"
# DATA_PATH = ROOT_DIR + \
#     "/../../data/catalogs/v7/THEx-assembled-v7.1a-mags-legacy-xcalib-minxcal.fits"
