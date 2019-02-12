import sys
import os
import numpy as np
from PIL import Image
import cv2
import glob

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
                labels.append(f.readline())

    return labels


def resizeImages(img_files):
        for i in range(len(img_files)):
            new_im = cv2.imread(img_files[i][1]) #replacing img paths with resized matrix 
            res = cv2.resize(new_im,dsize=(64,64))
            img_files[i][1] = res
        return img_files
        

    
def main():

    labels_dir = "other_images/Emotion/"
    labels = getLabels(labels_dir,".txt")


    imgs_dir = "other_images/cohn-kanade-images/"
    img_files = traverseFolders(imgs_dir,".png",labels)
    resized_images = resizeImages(img_files)

    for i in resized_images:
        print("i " + str(i))
        
    print("size of resized_images: " + str(len(resized_images)))
        


main()