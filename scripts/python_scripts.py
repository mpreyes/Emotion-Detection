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
            labels.append("no label")
        for f in files:
            if f.endswith(extension) and not f.startswith("."):
                f = open(os.path.join(root,f),"r")
                labels.append(int(float(f.readline().strip("\n"))))

    return labels


def resizeImages(img_files,x,y):
        #datagen = ImageDataGenerator(target_size=(64,64))
        for i in range(len(img_files)):
            new_im = cv2.imread(img_files[i][1]) #replacing img paths with resized matrix 
            #k_img = img_to_array(img_files[i][1]) 
            #print("kmg " + k_img)
            res = cv2.resize(new_im,dsize=(64,64))
            #res = res[...,::-1]
            #k_img = img_to_array(res) 
            #img_files[i][1] = res
            x.append(res)
            y.append(img_files[i][0])
        return img_files
        
    
def main():
    imgs_dir = "other_images/cohn-kanade-images/"
    labels_dir = "other_images/Emotion/"
    x = [] #inputs (images)
    y = [] #labels 
    # labels = getLabels(labels_dir,".txt")
    # img_files = traverseFolders(imgs_dir,".png",labels)
    # resized_images = resizeImages(img_files,x,y)
    img_path = 'other_images/cohn-kanade-images/S999/003/S999_003_00000055.png'
    img = image.load_img(img_path, target_size=(64, 64))
    img2 = image.img_to_array(img)
    #x = preprocess_input(x)
    x.append(np.asarray(img2))
    y.append(np.asarray(1))
    #y = to_categorical(y, 7).astype(np.float32)
    x = np.asarray(x)
    y = np.asarray(y)
    print(np.shape(x))
    print(np.shape(y))

    model = Sequential()
    model.add(Dense(units=64, activation='relu',input_shape=(64, 64, 3)))
    

# #     model = Sequential([
# #     Flatten(input_shape=(64, 64,3)),
# #     Dense(128, activation='softmax'),
# #     #layers.Dense(10, activation=tf.nn.softmax)
# # ])

#     model.add(Conv2D(32, (3, 3), input_shape=( 64, 64,3)))

    model.add(Dense(units=1, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy'])

    model.fit(x, y, epochs=5, batch_size=32)

    # model = VGG16(weights='imagenet', include_top=False)
   
    

    # #features = model.predict(x)
    # x = np.expand_dims(x, axis=0)
 

    # p =  model.fit(x, y, epochs=5, batch_size=32)

    # //print('features: ', features)
            



main()