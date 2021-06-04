# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:05:16 2021

@author: angelou
"""

import os
import sys

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (CSVLogger)
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from model import DCUNet
from generator_file import My_Generator,EndEvaluation
print(sys.version)
print(tf.__version__)
from os import walk
#tf.config.list_physical_devices('GPU')
#tf.config.gpu.set_per_process_memory_fraction(0.75)
#tf.config.gpu.set_per_process_memory_growth(True)

# prepare training and testing set
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

        jacard_viz = ((np.sum(intersection)+1.0)/(np.sum(union)+1.0))
        plt.suptitle('Jacard Index'+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard_viz))

        plt.savefig(f'results\\images\\epoch_{epoch+1:02d}\\i'+str(i)+'.png',format='png')
        plt.close()
    
    jacard_1 = 0
    jacard_2 = 0
    dice_1 = 0
    dice_2 = 0
    tversky_value=0
    smooth = 1.0
    alpha = 0.75

    for i in range(len(Y_test)):

        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection
        jacard_1 += ((np.sum(intersection)+smooth)/(np.sum(union)+smooth))

        true_pos = K.sum(y2 * yp_2)
        false_neg = K.sum(y2 * (1 - yp_2))
        false_pos = K.sum((1 - y2) * yp_2)

        tversky_value += (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

        dice_1 += (2. * np.sum(intersection) + smooth) / (np.sum(yp_2) + np.sum(y2) + smooth)

        jacard_2 += jacard(Y_test[i].astype('float32'),yp[i].astype('float32'))
        dice_2 += dice_coef(Y_test[i].astype('float32'),yp[i].astype('float32'))

    jacard_1 /= len(Y_test)
    dice_1 /= len(Y_test)
    jacard_2 /= len(Y_test)
    dice_2 /= len(Y_test)
    tversky_value /= len(Y_test)
    


    print('Jacard Index : '+str(jacard_1))
    print('Dice Coefficient : '+str(dice_1))

    fp = open('results/log.txt','a')
    fp.write(str(epoch+1)+','+str(jacard_1)+','+str(dice_1)+','+str(tversky_value.numpy())+','
             +str(jacard_2.numpy()) + ',' + str(dice_2.numpy())+'\n')
    fp.close()

    fp = open('results/best.txt','r')
    best = fp.read()
    fp.close()

    if(jacard_1>float(best)):
        print('***********************************************')
        print('Jacard Index improved from '+str(best)+' to '+str(jacard_1))
        print('***********************************************')
        fp = open('results/best.txt','w')
        fp.write(str(jacard_1))
        fp.close()
        model.save(f'results\\best.h5')

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
        base_model = DCUNet(height=height, width=width, channels=3)
    else:
        # this line is used if we're using checkpoints by epoch
        #base_model = load_model(f"{path}/ep_{checkpoint_epoch:02d}.h5",custom_objects={"focal_tversky":focal_tversky,"dice_coef":dice_coef,"jacard":jacard})
        base_model = load_model(f"{path}/checkpoint.h5",
                                custom_objects={"focal_tversky": focal_tversky, "dice_coef": dice_coef,
                                                "jacard": jacard})

    model = base_model
    return model
#'''
#starting params
PATH = "C:\\Users\\Boris\\Desktop\\DC-UNet-main\\DATASET"
CHECKPOINT_EP = 0
WORKERS = 2
batch_size = 4
EPOCHS = 500
HEIGHT = 96 #96
WIDTH=128 #128
X = []
Y = []

mask = os.listdir(f'{PATH}\\Ground Truth')
original = os.listdir(f'{PATH}\\Original')
mask_aug = os.listdir(f'{PATH}\\augmented_images\\labels')
original_aug = os.listdir(f'{PATH}\\augmented_images\\images')
aug_klase = []
for orig in original_aug:
    if 'benign' in orig:
        aug_klase.append('benign')
    elif 'malignant' in orig:
        aug_klase.append('malignant')
    elif 'normal' in orig:
        aug_klase.append('normal')
aug_df = pd.concat([pd.Series(original_aug),pd.Series(mask_aug),pd.Series(aug_klase)],axis=1)
klase = [f.split(' ')[0] for f in original]
df_origs = pd.concat([pd.Series(original),pd.Series(mask),pd.Series(klase)],axis=1)
df = pd.concat([df_origs,aug_df],axis=0)
x_train, x_test, y_train, y_test = train_test_split(df[0], df[1], test_size=0.2,
                                                      stratify=df[2], random_state=42)


#checkpoint = ModelCheckpoint(filepath="C:\\Users\\Boris\\Desktop\\DC-UNet-main\\models\\v1_ep_{epoch:02d}.hdf5", monitor="val_loss", verbose=1,
#                             save_best_only=False, mode='min', save_weights_only=False)
#reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto',
#                                   min_delta=0.0001)
#early = EarlyStopping(monitor="val_loss", mode="min", patience=9)

train_generator = My_Generator(x_train, y_train, PATH, batch_size=batch_size, is_train=True,IMG_size=(WIDTH,HEIGHT)) #accurate order of width and height
train_mixup = My_Generator(x_train, y_train, PATH, batch_size=batch_size, is_train=True, mix=False, augment=False,IMG_size=(WIDTH,HEIGHT))
valid_generator = My_Generator(x_test, y_test, PATH, batch_size=batch_size, is_train=False,IMG_size=(WIDTH,HEIGHT))
#end_eval = EndEvaluation(validation_data=(valid_generator, y_test),raw_data = (x_test.iloc[:10,:],y_test.iloc[:10,:]), batch_size=batch_size, interval=1)
model_checkpoint= tf.keras.callbacks.ModelCheckpoint(
    'C:\\Users\\Boris\\Desktop\\DC-UNet-main\\results\\weights_best', monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None)
csv_logger = CSVLogger(filename='C:\\Users\\Boris\\Desktop\\DC-UNet-main\\results\\v1_training_log.csv', separator=',', append=True)
callbacks_list = [csv_logger,model_checkpoint]#,end_eval
# different loss functions
#model = DCUNet(height=192, width=256, channels=3)
model = create_model((HEIGHT,WIDTH),checkpoint_epoch=CHECKPOINT_EP) # (height,width)
model.compile(optimizer=Adam(1e-5), loss=focal_tversky, metrics=[dice_coef, jacard, tversky, 'accuracy'])
#tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#binary_crossentropy
#model.summary()
#saveModel(model)

fp = open('results/log.txt','w')
fp.close()
fp = open('results/best.txt','w')
fp.write('-1.0')
fp.close()
    
#trainStep(model, X_train, Y_train, X_test, Y_test, epochs=300, batchSize=2,checkpoint_epoch=CHECKPOINT_EP)

model.fit(
    train_mixup,
    steps_per_epoch=int(np.ceil(float(len(x_train)) / float(batch_size))),
    validation_data=valid_generator,
    validation_steps=int(np.ceil(float(len(x_test)) / float(batch_size))),
    epochs=EPOCHS,
    verbose=1,max_queue_size=1,
    workers=1, use_multiprocessing=False ,initial_epoch=0,callbacks=callbacks_list)
fp = open('results/opis.txt','w')
fp.write('focal_tversky,96x128')
fp.close()

