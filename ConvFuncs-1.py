#Imports
import os
import pandas as pd
import plotly.express as px
from preprocessors import AspectAwarePreprocessor
import jobsConfig as config
import numpy as np
import h5py
import cv2
import json
import sys

#=======================================================================================================================================================================================================================================#

def label_distributions(raw_data, hist = True):
    raw_data_paths = pd.DataFrame([os.path.join(raw_data, file_) for file_ in os.listdir(raw_data)], columns = ["path"])
    
    #Print number of datapoints
    print("Number of datapoints: {}".format(raw_data_paths.shape[0]))
    
    #Get labels
    labels = [path.split("_")[-1].split(".")[0] for path in raw_data_paths["path"]]
    raw_data_paths["label"] = labels
    
    #Show value_counts
    print("Value counts:\n{}".format(raw_data_paths["label"].value_counts()))
    
    if hist:
        #Display histogram
        fig = px.histogram(raw_data_paths, x = "label")
        fig.update_layout(
        title_text = "Label Distributions",
        title = {"x":0.5},
        xaxis_title_text = "Value",
        yaxis_title_text = "Count",)
        fig.show(config = {"displayModeBar":False})

    #Return the containing the paths of each image the label for each image
    return raw_data_paths["path"].values, raw_data_paths["label"].values

#=======================================================================================================================================================================================================================================#

def label_to_numeric(y):
    return np.array([0 if label == "DEFECTIVE" else 1 for label in y]).astype("int") #Force integer output

#=======================================================================================================================================================================================================================================#

class HDF5DatasetWriter():
        
    def __init__(self, outputPath, dims, dataKey = "images", bufSize = 1000):
    
        #Check if output path exists so it is not overwritten
        if os.path.exists(outputPath):
            proceede_flag = input("The supplied database {} already exists. Would you like to delete the database and proceede (y/n)?: ".format(outputPath))
            if proceede_flag == "y":
                pass
            if proceede_flag == "n":
                sys.exit("Database already exists")


        #Open up the database and create two datasets, one for images/features and another for labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype = "float") #Create a dataset within database to hold data
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype = "int") #Create dataset within database to hold labels
        self.bufSize = bufSize
        self.buffer = {"data":[], "labels":[]}
        self.idx = 0

    def add(self, rows, labels):

        #Add rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        #Check to see if buffer needs flushed
        if len(self.buffer["data"]) > self.bufSize:
            self.flush()

    def flush(self):

        #Write buffer contents to disk, then reset buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i #Update index
        self.buffer = {"data":[], "labels":[]} #Reset buffer
        
    def store_class_labels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype = dt)
        labelSet[:] = classLabels
        
    def close(self):
        
        #Check buffer then close
        if len(self.buffer["data"]) > 0:
            self.flush()
            
        #Close database
        self.db.close()

#=======================================================================================================================================================================================================================================#

def create_HDF5_dataset(writer, dataset, target_shape, output = config.DATASET_MEAN):

    #Initializations
    aap = AspectAwarePreprocessor(target_shape[0], target_shape[1])

    #Initialize lists for coloi values if we are creating training set
    if dataset["dType"] == "train":
        R, G, B = ([], [], [])

    for path, label in zip(dataset["paths"], dataset["labels"]):

        image = cv2.imread(path)
        image = aap.preprocess(image)

        #Check to see if RBG arrays need updated
        if dataset["dType"] == "train":
            (b,g,r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add([image], [label])

    #Close the writer
    writer.close()

    #Update mean color values if we are creating training set
    if dataset["dType"] == "train":
        D = {"R":np.mean(R), "G":np.mean(G), "B":np.mean(B)}
        f = open(output, "w")
        f.write(json.dumps(D))
        f.close()

#=======================================================================================================================================================================================================================================#