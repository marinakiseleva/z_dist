#orig_dir = model.dir
import os 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
new_dir = ROOT_DIR + "/figures/evaluation/"

LSST_dir = new_dir + "lsst_test/"

os.mkdir(LSST_dir)
