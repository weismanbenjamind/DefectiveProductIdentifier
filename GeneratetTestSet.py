"""
Script to label test data and create HDF5 database for testing data
"""

#=======================================================================================================================================================================================================================================#

#Imports
import jobsConfig as config
from preprocessors import label_data
from ConvFuncs import label_distributions, label_to_numeric, create_HDF5_dataset, HDF5DatasetWriter
import os
import argparse

#=======================================================================================================================================================================================================================================#

#Clear the screen
os.system("cls")

#Arguement Parser
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--raw_data", default = config.RAW_TEST_DATA, help = "File path to raw data")
ap.add_argument("-tro", "--test_output", default = config.TEST_HDF5, help = "Output for .hdf5 file associated with test set")
args = vars(ap.parse_args())

#First Label Data
print("[INFO] Labelling Data")
label_data(config.UNLABELLED_TEST_DATA, args["raw_data"])

#Check label distributions and get paths with corresponding labels
print("[INFO] Analyzing data distributions\n")
paths, labels = label_distributions(args["raw_data"], hist = False)

#Encode labels
labels = label_to_numeric(labels)

#Create a dictionary for dataset then create the .h5py database
dataset = {"dType":"test", "paths":paths, "labels":labels, "output":args["test_output"]}
print("[INFO] Creating database {}".format(dataset["output"]))
create_HDF5_dataset(HDF5DatasetWriter(dataset["output"], (len(dataset["labels"]),256,256,3)),dataset,(256,256))

#=======================================================================================================================================================================================================================================#