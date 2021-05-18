# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:05:16 2021

@author: angelou
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (CSVLogger)
from tensorflow.keras.models import load_model

from model import DCUNet

print(sys.version)
print(tf.__version__)
#tf.config.list_physical_devices('GPU')
#tf.config.gpu.set_per_process_memory_fraction(0.75)
#tf.config.gpu.set_per_process_memory_growth(True)

# prepare training and testing set

X = []
Y = []
PATH = "C:\\Users\\Boris\\Desktop\\DC-UNet-main\\DATASET"
for file in os.listdir(PATH+"\\Original"):
    path = PATH + "\\Original\\" + file
    img = cv2.imread(path, 1)
    resized_img = cv2.resize(img, (128, 96), interpolation=cv2.INTER_CUBIC)

    X.append(resized_img)

    path2 = PATH + "\\Ground Truth\\" + file[:-4]+"_mask.png"
    msk = cv2.imread(path2, 0)

    resized_msk = cv2.resize(msk, (128, 96), interpolation=cv2.INTER_CUBIC)

    Y.append(resized_msk)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))


X_train = X_train / 255
X_test = X_test / 255
Y_train = Y_train / 255
Y_test = Y_test / 255

Y_train = np.round(Y_train,0)	
Y_test = np.round(Y_test,0)	

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
#checkpoint = ModelCheckpoint(filepath="C:\\Users\\Boris\\Desktop\\DC-UNet-main\\models\\v1_ep_{epoch:02d}.hdf5", monitor="val_loss", verbose=1,
#                             save_best_only=False, mode='min', save_weights_only=False)
#reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto',
#                                   min_delta=0.0001)
#early = EarlyStopping(monitor="val_loss", mode="min", patience=9)
csv_logger = CSVLogger(filename='C:\\Users\\Boris\\Desktop\\DC-UNet-main\\results\\v1_training_log.csv', separator=',', append=True)
callbacks_list = [csv_logger]
# different loss functions
def dice_coef(y_true, y_pred):
    smooth = 1.0  #0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union

def dice_coef_loss(y_true,y_pred):
    return 1 - dice_coef(y_true,y_pred)

def iou_loss(y_true,y_pred):
    return 1 - jacard(y_true, y_pred)

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.75
    smooth = 1
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

# training

def saveModel(model,epoch):

    model_json = model.to_json()

    try:
        os.makedirs('models')
    except:
        pass
    
    fp = open('models/modelP.json','w')
    fp.write(model_json)
    # this line is used if we're using checkpoints by epoch
    #model.save(f'models\\working_models\\ep_{epoch+1:02d}.h5')
    model.save(f'models\\working_models\\checkpoint.h5')


def evaluateModel(model, X_test, Y_test, batchSize,epoch):
    
    try:
        os.makedirs('results')
    except:
        pass 
    
    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    yp = np.round(yp,0)

    try:
        os.makedirs(f'results\\images\\epoch_{epoch+1:02d}')
    except:
        pass

    for i in range(10):

        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(X_test[i])
        plt.title('Input')
        plt.subplot(1,3,2)
        plt.imshow(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]))
        plt.title('Prediction')

        intersection = yp[i].ravel() * Y_test[i].ravel()
        union = yp[i].ravel() + Y_test[i].ravel() - intersection

        jacard = ((np.sum(intersection)+1.0)/(np.sum(union)+1.0))
        plt.suptitle('Jacard Index'+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard))

        plt.savefig(f'results\\images\\epoch_{epoch+1:02d}\\i'+str(i)+'.png',format='png')
        plt.close()
    
    jacard = 0
    dice = 0
    tversky_value=0
    smooth = 1.0
    alpha = 0.75

    for i in range(len(Y_test)):

        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection
        jacard += ((np.sum(intersection)+smooth)/(np.sum(union)+smooth))

        true_pos = K.sum(y2 * yp_2)
        false_neg = K.sum(y2 * (1 - yp_2))
        false_pos = K.sum((1 - y2) * yp_2)

        tversky_value += (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

        dice += (2. * np.sum(intersection) + smooth) / (np.sum(yp_2) + np.sum(y2) + smooth)

    
    jacard /= len(Y_test)
    dice /= len(Y_test)
    tversky_value /= len(Y_test)
    


    print('Jacard Index : '+str(jacard))
    print('Dice Coefficient : '+str(dice))

    fp = open('models/log.txt','a')
    fp.write(str(epoch+1)+','+str(jacard)+','+str(dice)+','+str(tversky_value.numpy())+'\n')
    fp.close()

    fp = open('models/best.txt','r')
    best = fp.read()
    fp.close()

    if(jacard>float(best)):
        print('***********************************************')
        print('Jacard Index improved from '+str(best)+' to '+str(jacard))
        print('***********************************************')
        fp = open('models/best.txt','w')
        fp.write(str(jacard))
        fp.close()
        model.save(f'models\\best.h5')

    saveModel(model,epoch=epoch)


def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize,checkpoint_epoch=0):

    
    for epoch in range(checkpoint_epoch,epochs):
        print('Epoch : {}'.format(epoch+1))
        model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=1,callbacks=callbacks_list)

        evaluateModel(model,X_test, Y_test,batchSize,epoch)

    return model 

def create_model(input_shape, checkpoint_epoch=0):
    height,width = input_shape
    path = "C:/Users/Boris/Desktop/DC-UNet-main/models/working_models"
    if checkpoint_epoch==0:
        base_model = DCUNet(height=96, width=128, channels=3)
    else:
        # this line is used if we're using checkpoints by epoch
        #base_model = load_model(f"{path}/ep_{checkpoint_epoch:02d}.h5",custom_objects={"focal_tversky":focal_tversky,"dice_coef":dice_coef,"jacard":jacard})
        base_model = load_model(f"{path}/checkpoint.h5",
                                custom_objects={"focal_tversky": focal_tversky, "dice_coef": dice_coef,
                                                "jacard": jacard})
    model = base_model

    return model

CHECKPOINT_EP = 0
#model = DCUNet(height=192, width=256, channels=3)
model = create_model((157,187),checkpoint_epoch=CHECKPOINT_EP)
model.compile(optimizer='adam', loss=iou_loss, metrics=[dice_coef, jacard, 'accuracy'])
#binary_crossentropy
#model.summary()
#saveModel(model)

fp = open('models/log.txt','w')
fp.close()
fp = open('models/best.txt','w')
fp.write('-1.0')
fp.close()
    
trainStep(model, X_train, Y_train, X_test, Y_test, epochs=300, batchSize=2,checkpoint_epoch=CHECKPOINT_EP)