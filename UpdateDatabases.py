#Script to update databases with new data
#For the time being the database will be deleted then recreated as new data is added

#=================================================================================================================================================================================================================#

#Imports
import jobsConfig as config
from preprocessors import label_data, AspectAwarePreprocessor
from ConvFuncs import label_distributions, label_to_numeric, HDF5DatasetWriter, create_HDF5_dataset
from sklearn.model_selection import train_test_split
import os
import argparse

#=================================================================================================================================================================================================================#
#Clear the screen
os.system("cls")

#Arguement Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default = config.RAW_DATA, help = "File path to raw data")
ap.add_argument("-t", "--test", default = False, type = bool, help = "Should a test set also be generated?")
ap.add_argument("-to", "--train_output", default = config.TRAIN_HDF5, help = "Output for .hdf5 file associated with training set")
ap.add_argument("-vo", "--validation_output", default = config.VAL_HDF5, help = "Output for .hdf5 file associated with validation set")
ap.add_argument("-tro", "--test_output", default = config.TEST_HDF5, help = "Output for .hdf5 file associated with test set")
args = vars(ap.parse_args())

#Fist label the data and get it moved to the raw data folder
print("[INFO] Labelling Data")
label_data(config.UNLABELLED_DATASET, args["input"])

#Check label distributions and get paths with corresponding labels
print("[INFO] Analyzing data distributions\n")
paths, labels = label_distributions(args["input"], hist = False)

#Split up data and get ready to create .h5py datasets
print("\n[INFO] Splitting data into training, validation, and testing sets")

if not args["test"]:
    X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=config.VAL_IMAGES_RATIO, stratify=labels, random_state=42)

    print("\nNumber of training instances: {}\nNumber of validation instances: {}\n".format(len(y_train), len(y_val)))

    #Enode categorical variables
    labels = [label_to_numeric(y) for y in (y_train, y_val)]

    #Create list of dictionaries for datasets
    datasets = [
        {"dType":"train", "paths":X_train, "labels":labels[0], "output":args["train_output"]},
        {"dType":"val", "paths":X_val, "labels":labels[1], "output":args["validation_output"]},
    ]

else:
    X_train, X_test, y_train, y_test = train_test_split(paths, labels, test_size=config.TEST_IMAGES_RATIO, stratify=labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config.VAL_IMAGES_RATIO, stratify=y_train, random_state=42)
    
    #Indicate number of instances per dataset
    print("\nNumber of training instances: {}\nNumber of validation instances: {}\nNumber of testing instances: {}\n".format(len(y_train), len(y_val), len(y_test)))

    #Enode categorical variables
    labels = [label_to_numeric(y) for y in (y_train, y_val, y_test)]

    #Create list of dictionaries for datasets
    datasets = [
        {"dType":"train", "paths":X_train, "labels":labels[0], "output":args["train_output"]},
        {"dType":"val", "paths":X_val, "labels":labels[1], "output":args["validation_output"]},
        {"dType":"test", "paths":X_test, "labels":labels[2], "output":args["test_output"]},
    ]

#Write data to datasets
for dataset in datasets:
    print("[INFO] Creating database {}".format(dataset["output"]))
    create_HDF5_dataset(HDF5DatasetWriter(dataset["output"], (len(dataset["labels"]),256,256,3)),dataset,(256,256))