#=======================================================================================================================================================================================================================================#

#Imports
import os
import numpy as np
import pandas as pd

#=======================================================================================================================================================================================================================================#

def get_paths(root, reading_types):
    """Function to grab all .csv files containing a certain reading type (e.g. keyword).

    Args:
        root (str): Parent directory which contains all .csv files pertaining to readings.
        reading_types (list of strings): List where each element is a string which indicates the reading type (keyword) found in the .csv files pertaining to the reading type is question.

    Returns:
        list: List where each list element is a list containing all the .csv files for each respective keyword.
    """

    #Initialize list to hold readings
    readings = []

    #Loop through reading types and get all paths of relevant readings
    for reading in reading_types:
        readings.append([os.path.join(root, file) for file in os.listdir(root) if reading in file])

    #Return list of paths
    return readings

#=======================================================================================================================================================================================================================================#

def get_reading(paths):
    """Function to read in a list of depth readings as .csv files and average the depth arrays to make one average depth array.

    Args:
        paths (list of strings): List of file paths pertaining to depth readings represented as .csv files.

    Returns:
        np.array: Array representing the average of all input depth reading arrays (.csv files).
    """

    readings = []

    #Read all paths
    for path in paths:
        readings.append(pd.read_csv(path).values)
    
    #Average readings and return
    return np.mean(np.array(readings), axis = 0)

#=======================================================================================================================================================================================================================================#