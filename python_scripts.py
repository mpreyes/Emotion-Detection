import sys
import os
import numpy as np
from PIL import Image
import cv2


def traverseFolders(rootdir):
    img_paths = []
    for subdir, dirs, images, in os.walk(rootdir):
        for img in images:
            if img.endswith("png"):
                img_paths.append(os.path.join(subdir,img))
            #print("s " + str(subdir))
            #print("i " + str(img))
            #convertImage(img)
    for i in range(10):
        #print("i " + str(i))
        convertImage(img_paths[i])
            

def convertImage(filename):
        img = Image.open(filename)
        arr = np.array(img)
        print(img)
        #print(arr)
        print(img.format, img.size, img.mode)
        new_im = cv2.imread(filename)
        res = cv2.resize(new_im,dsize=(64,64),interpolation=cv2.INTER_CUBIC)
        img.show()
        print(res)

    


def main():
    rootdir = "other_files/cohn-kanade-images/"
    traverseFolders(rootdir)



main()