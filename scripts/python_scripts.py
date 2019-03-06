import sys
import os
import numpy as np
from PIL import Image
import cv2
import glob
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils.np_utils import to_categorical

#create dictionary with filepath, label
#training data -> labeled

#divide img by 255
#shuffle images randomly = True
#use unlabed images as training data
#VGGNet16, ResNet, VGGNet19

def traverseFolders(rootdir,extension,labels):
    labeled_data = []
    test_data = []
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
                if f.endswith(extension) and not f.startswith("."): #-1 is if the sequence is not labeled
                    temp_arr.append(labels[files_seq])
                    temp_arr.append(os.path.join(i,f))
                    if labels[files_seq] != -1:
                        labeled_data.append(temp_arr)
                    elif labels[files_seq] == -1:
                         test_data.append(temp_arr)
                   
        files_seq += 1

    print("test " + str(len(test_data)))
    print("label " + str(len(labeled_data)))
    return labeled_data, test_data
        


def getLabels(rootdir,extension): #grabbing labels from labels folder
    labels = []
    for root, dirs, files, in os.walk(rootdir):
        if len(files) != 0: 
            labels.append(-1) #meaning no label was assigned for this sequence of images 
            for f in files:
                if f.endswith(extension) and not f.startswith("."):
                    f = open(os.path.join(root,f),"r")
                    labels.append(int(float(f.readline().strip("\n"))))
    return labels


def resizeImages(img_files,x,y):
        for i in range(len(img_files)): 
            img = image.load_img(img_files[i][1], target_size=(64, 64))
            resized_img = image.img_to_array(img) 
            x.append(resized_img)
            y.append(img_files[i][0])
        
        


def inputDataSummary(inputs,labels):
    print(np.shape(inputs))
    print(np.shape(labels))
    #print(inputs)
    #print(labels)
    

def sequentialModel():
    model = Sequential()
    model.add(Flatten(input_shape=(64, 64, 3)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=8, activation='softmax')) 
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    return model

#increase img size 224 * 224, scale and divide by 256
def conv2DModel():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu',data_format = "channels_last",input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def runModel(model, inputs, labels):
    model.fit(inputs, labels, epochs=30, batch_size=32) #TODO: set to 500
    model.summary()
    score = model.evaluate(inputs, labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])



def main():
    imgs_dir = "../inputs/cohn-kanade-images/"
    labels_dir = "../inputs/Emotion/"
    # imgs_dir = "../subset_images/cohn-kanade-images/"
    # labels_dir = "../subset_images/Emotion/"
    x = [] #inputs (images)
    y = [] #labels 
    labels = getLabels(labels_dir,".txt")
    labeled_data,test_data = traverseFolders(imgs_dir,".png",labels)

    resizeImages(labeled_data,x,y)
    print(len(labeled_data) +  len(test_data))

    inputs = np.asarray(x) / float(255)
    labels = np.asarray(y)
    labels = to_categorical(labels)
    inputDataSummary(x,y)
    conv2D = conv2DModel()
    #vgg16model = VGG16()

    #runModel(conv2D,inputs,labels)
    #runModel(vgg16model,inputs,labels)

    
    # print(vgg16model.summary())

    #print("prediction" + conv2D.predict(np.array(x[0])))
    


main()