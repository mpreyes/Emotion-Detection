

import sys
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape, Dropout
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image as keras_image
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.models import Model, model_from_json
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt




IMAGES_PATH = "../inputs/images/"
LABELS_PATH = "../inputs/labels/"

WEIGHTS_PATH = "../outputs/weights/"
NUM_CLASSES  = 8 


MODELS = ['vgg16', 'simple']

def read_inputs():
    X = []
    Y = []
    for root, dirs, files, in os.walk(LABELS_PATH):
        for file in files:
            if file.endswith(".txt") and not file.startswith("."):
                images_path = root.replace(LABELS_PATH, IMAGES_PATH)
                images      = list(filter(lambda x: not x.startswith("."), os.listdir(images_path))) # keep images that doesn't start with .

                label_file = open(os.path.join(root,file),"r")
                label = int(float(label_file.readline().strip("\n")))

                images.sort() # only get the last four (maybe ?)
                for img in images[-4:]:
                    tmp = keras_image.load_img(images_path + "/" + img , target_size=(64, 64))
                    tmp = keras_image.img_to_array(tmp) / float(255)
                    X.append(tmp)
                    Y.append(label)
    return np.asarray(X), np_utils.to_categorical(np.asarray(Y), NUM_CLASSES)




def simple_model(): #pass in path to weights
    model = Sequential()
    model.add(Flatten(input_shape=(64, 64, 3)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


def not_my_model(model):
    stacked = Flatten()(model.output)
    stacked = Dense(64, activation='relu')(stacked)
    predictions = Dense(8, activation='softmax')(stacked)
    model = Model(input=model.input, output=predictions)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model  


def our_model(): #add dropout layers, 
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu',data_format = "channels_last",input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Dropout(0.01))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Flatten())

    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model



def runModel(model, model_name, X, Y):
    filepath="../saved/" + model_name +"/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1,period=1)
    callbacks_list = [checkpoint] # class_weight=class_weights
#    label_integers = np.argmax(labels, axis = 1)
#    class_weights = class_weight.compute_class_weight('balanced',np.unique(label_integers),label_integers)
    history = model.fit(X, Y,validation_split=0.15, verbose=1, epochs=200, batch_size=24, shuffle=True, callbacks=callbacks_list) #TODO: set to 500
    return history


def getModelAcc(model, X,Y):
    print(model.summary())
    score = model.evaluate(X, Y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def saveModel(model,model_name):
    model_json = model.to_json()
    with open("../saved/" + model_name + "/"+ model_name + "_end.json","w") as json_file:
        json_file.write(model_json)  
    model.save_weights("../saved/" + model_name + "/"+ model_name + "_end.h5")
    print("saving model...")


def loadModel(model_name):
    model_folder = "../saved/" + model_name + "/"
    json_file = open(model_folder + model_name + "_end.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights( model_folder + model_name + "_end.h5")
    print("loading model and recompiling...")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    loaded_model._make_predict_function()

    return loaded_model





#def batch_predictions(loaded_model,X,Y):
#    print("making predictions ")
##  
#    correctly_pred = 0
#    incorrectly_pred = 0
#    
#    for i in range(100):
#        print("actual " + str(X[i]) + " " + str(Y[i]), end =" ")
#        resized_img = np.expand_dims(X[i], axis=0)
#        preds = loaded_model.predict(resized_img,verbose=0)
#        print("predicted " + str(preds),end=" ")
#        pred_int = sort(preds)
#        p =  pred_int[0]
#        if Y[i] == preds:
#            correctly_pred += 1
#        else:
#            incorrectly_pred += 1
#        print("\n")
#    print("total" + str(len(X)) + " correctly predicted: " + str(correctly_pred) + " incorrectly predicted: " + str(incorrectly_pred))
#
#
#
#

def generateConfusionMatrix(loadedModel, X,Y):
    y_true = []
    y_pred = []
    
    Y = Y.argmax(1)
    for i in range(len(Y)):
        y_true.append(Y[i])
        resized_img = np.expand_dims(X[i], axis=0)
        p = loadedModel.predict(resized_img) #array of preds
        ps = p.argmax(1)
        y_pred.append(ps)
    return confusion_matrix(y_true, y_pred, normalize=True)


def generatePlots(history):
    seed = 7
    np.random.seed(seed)
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    





def main():
    X, Y = read_inputs()
#    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape = (64,64,3)) 
#    vgg19 = VGG19(weights='imagenet', include_top=False, input_shape = (64,64,3)) 
#    resNet = ResNet50(weights='imagenet', include_top=False, input_shape = (64,64,3)) 
#    
#    
#    notMyVgg16Model = not_my_model(vgg16)
#    notMyVgg19Model = not_my_model(vgg19)
#    notMyResNetModel = not_my_model(resNet)
#    simpleModel = simple_model()
#    ourModel = our_model()
    
#    print("vgg16 Model ")
#    vgg16History = runModel(notMyVgg16Model,"vgg16",X,Y) 
#    saveModel(notMyVgg16Model,"vgg16")
#    print("vgg19 Model ")
#    vgg19History = runModel(notMyVgg19Model,"vgg19",X,Y) 
#    saveModel(notMyVgg19Model,"vgg19")
#    print("resNet Model ")
#    resNetHistory = runModel(notMyResNetModel,"resNet",X,Y) 
#    saveModel(notMyResNetModel,"resNet")
#    print("our Model ")
#    ourModelHistory = runModel(ourModel,"conv2D",X,Y) 
#    saveModel(ourModel,"conv2D")
#    
#    generatePlots(vgg16History)
#    generatePlots(vgg19History)
#    generatePlots(resNetHistory)
#    generatePlots(ourModelHistory)
#
#    
    vgg16Loaded = loadModel("vgg16")
    vgg19Loaded = loadModel("vgg19")
    resNetLoaded = loadModel("resNet")
    ourModelLoaded = loadModel("conv2D")
    
#    getModelAcc(vgg16Loaded,X,Y)
#    getModelAcc(vgg19Loaded,X,Y)
#    getModelAcc(resNetLoaded,X,Y)
#    getModelAcc(ourModelLoaded,X,Y)
    
#    
    vggconfusionMatrix = generateConfusionMatrix(vgg16Loaded, X,Y)
#    vvg19ConfusionMatrix = generateConfusionMatrix(vgg16Loaded, X,Y)
#    resNetConfusionMatrix = generateConfusionMatrix(resNetLoaded, X,Y)
    ourModelConfusionMatrix = generateConfusionMatrix(ourModelLoaded, X,Y)
#    
    print(vggconfusionMatrix)
    print("ours")
    print(ourModelConfusionMatrix)
#    batch_predictions(vgg16Loaded,X,Y)
    
        #checkpoint = ModelCheckpoint(WEIGHTS_PATH + "vgg16-weights.h5", verbose=1, save_best_only=True)
        #callbacks_list = [checkpoint]
    
        
        #plt.imshow(X[0])
#        preds = model.predict(X[:10])
        #print(preds[0])
        #print(Y[0])

main()