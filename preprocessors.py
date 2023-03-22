#File to store preprocessors for jobs project

#=======================================================================================================================================================================================================================================#

#Imports
import jobsConfig as config
import os
import cv2
import imutils

#=======================================================================================================================================================================================================================================#

def label_data(input_dataset, output_dataset, rotation_key = cv2.ROTATE_90_CLOCKWISE, target_height = 4032):
    """Function which rotates images in the input dataset to a target height (if needed), saves these images to a new directory, and deletes the image from the input directory.
        Assumes the target height will be one of the dimensions of the input image. 

    Args:
        input_dataset (str, optional): Path to input images which may or may not need rotated. Defaults to config.PREPROCESS_DATASET.
        output_dataset (str, optional): Path to output dataset where images will be saved. Defaults to config.RAW_DATA.
        rotation_key (cv2 class variable, optional): Rotation key to be used in cv2.ROTATE_90_CLOCKWISE. Defaults to cv2.ROTATE_90_CLOCKWISE.
        target_height (int, optional): Target height (px) of images. Assumed to be one of the dimensions of the images. Defaults to 4032.
    """

    #Grab number of images
    image_names = os.listdir(input_dataset)
    num_images = len(image_names)

    #Loop through all pictures in input directory
    for i, name in enumerate(image_names):
        img_flag = True #Flag for reading images

        #Get paths and read in image
        try:
            print("[INFO] Processing image {}/{}".format(i+1, num_images))
            path = os.path.join(input_dataset, name)
            img = cv2.imread(path)
        except:
            print("Error: could not read image {}".format(path))
            img_flag = False #Set flag to false and ignore this image

        if img_flag: #If our flag is true, process the image

            #Save the image to the raw data file, rotating the image if need be
            if img.shape[0] != target_height:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            #Only grab images which are not labelled
            #if img.split("_")[-2] != "LABELLED":

            #Update user, display image, and get user input
            cv2.namedWindow("Insulation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Insulation", 600, 600)
            cv2.moveWindow("Insulation", 100, 100)
            cv2.imshow("Insulation", img)
            key = cv2.waitKey(0)
            
            #Kill all open windows
            cv2.destroyAllWindows()

            #Label the image depending on the key pressed
            if key == ord("1"):
                cv2.imwrite(os.path.join(output_dataset, name.split("_")[1].split(".")[0] + "_LABELLED_SATISFACTORY.jpg"), img)
                #Remove the image from the input dataset
                os.remove(path)
            elif key == ord("2"):
                cv2.imwrite(os.path.join(output_dataset, name.split("_")[1].split(".")[0] + "_LABELLED_DEFECTIVE.jpg"), img)
                #Remove the image from the input dataset
                os.remove(path)
            elif key == ord("0"):
                os.remove(path)
            else:
                print("Error, invalid key pushed. {} not processed".format(path))

#=======================================================================================================================================================================================================================================#

class AspectAwarePreprocessor:

    def __init__(self, width, height, inter = cv2.INTER_AREA):
        #Store target width, height, and interpolation method
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        #Grab dims of image in question
        h,w = image.shape[:2]
        dW=0
        dH=0

        #Crop along the smaller dimension
        if w < h:
            image = imutils.resize(image, width = self.width, inter = self.inter)
            dH = int((image.shape[0] - self.height)/2.0)
        else:
            image = imutils.resize(image, height = self.height, inter = self.inter)
            dW = int((image.shape[1] - self.width)/2.0)

        #Now that images have been resized, re-grab dimensions and perform the crop
        h,w = image.shape[:2]
        image = image[dH:h-dH, dW:w-dW]

        #Return a resized image to eliminate any roudning errors
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
#=======================================================================================================================================================================================================================================#
