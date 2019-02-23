import sys
import os
import numpy as np
from PIL import Image
import cv2
import glob
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils.np_utils import to_categorical

#create dictionary with filepath, label
#training data -> labeled

def traverseFolders(rootdir,extension,labels):

    labeled_data = []
    list_subdirs = [] #should be 123 "outer dirs in cohn-kanade-imgs"
    for root, dirs, files, in os.walk(rootdir):
        for d in dirs:
            if not d.startswith("S"): #grabbing the inner dirs, should be 593
                list_subdirs.append(os.path.join(root,d))

    files_seq = 0
    for i in list_subdirs:
        for rootdir, dirs, files in os.walk(i): #traversing through subdirs to grab files
            for f in files:
                temp_arr = []
                if f.endswith(extension) and not f.startswith("."): 
                    temp_arr.append(labels[files_seq])
                    temp_arr.append(os.path.join(i,f))
                    labeled_data.append(temp_arr)
        files_seq += 1

    return labeled_data
        


def getLabels(rootdir,extension): #grabbing labels from labels folder
    labels = []
    for root, dirs, files, in os.walk(rootdir):
        if len(files) == 0:
            labels.append(-1) #meaning no label was assigned for this sequence of images 
        for f in files:
            if f.endswith(extension) and not f.startswith("."):
                f = open(os.path.join(root,f),"r")
                labels.append(int(float(f.readline().strip("\n"))))

    return labels


def resizeImages(img_files,x,y):
        for i in range(1,15): #len(img_files)
            img = image.load_img(img_files[i][1], target_size=(64, 64))
            resized_img = image.img_to_array(img)
            x.append(resized_img)
            y.append(img_files[i][0])



def sequentialModel():
    model = Sequential()
    model.add(Dense(units=64, activation='relu',input_shape=(64, 64, 3)))
    model.add(Dense(units=7, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy'])
    return model




    
def main():
    imgs_dir = "../subset_images/cohn-kanade-images/"
    labels_dir = "../subset_images/Emotion/"
    x = [] #inputs (images)
    y = [] #labels 
    labels = getLabels(labels_dir,".txt")
    img_files = traverseFolders(imgs_dir,".png",labels)
    resizeImages(img_files,x,y)
    #x = preprocess_input(x)
    inputs = np.asarray(x)
    labels = np.asarray(y)
    labels = to_categorical(labels)
    #print("to cat " + str(labels))
    labels = np.expand_dims(labels, axis=0)
    labels = np.expand_dims(labels, axis=1)
    print(np.shape(inputs))
    print(np.shape(labels))
    seq_model = sequentialModel()

    seq_model.fit(inputs, labels, epochs=5, batch_size=32)
    


main()