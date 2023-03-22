#Configuration file for project jobs

#Path to full dataset, not split into testing and training sets, not labelled
RAW_DATA = "../data/RawData"

#Path to dataset of images which need preprocessed (rotated)
UNLABELLED_DATASET = "../data/ToLabel"

#Test set information
RAW_TEST_DATA = "../data/Test/RawData"
UNLABELLED_TEST_DATA = "../data/Test/ToLabel"

#How data will be split - update when the exact number of raw images is known
NUM_CLASSES = 2
VAL_IMAGES_RATIO = 0.25
TEST_IMAGES_RATIO = 0.25

#Where the HDF5 files for each dataset will be saved
TRAIN_HDF5 = "../data/hdf5/train.hdf5"
VAL_HDF5 =  "../data/hdf5/val.hdf5"
TEST_HDF5 =  "../data/hdf5/test.hdf5"
TRAIN_FEATURES = "../data/hdf5/train_ResNet50.hdf5"
VAL_FEATURES = "../data/hdf5/val_ResNet50.hdf5"
TEST_FEATURES = "../data/hdf5/test_ResNet50.hdf5"


#Where to save logistic regressor which predicts of ResNet50
MODEL_PATH = "output/colorstrom.pkl"

#Where to save dataset mean
DATASET_MEAN = "output/insulation_mean.json"

#Where to output files, plots, data, reports, etc
OUTPUT = "../output"

#Image dimensions
DIMENSIONS = (3024,4032)

#Where to store results
VAL_RESULTS = "../output/ValResults.csv"