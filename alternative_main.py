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
from generator_file import My_Generator,SaveImages,ValidPredict
from evaluation import tversky_loss,tversky,focal_tversky,jacard,dice_coef,iou_loss,dice_coef_loss
from evaluation import load_files,load_model,make_list,class_vector
from second_model import unet_model
print(sys.version)
print(tf.__version__)
from os import walk
#tf.config.list_physical_devices('GPU')
#tf.config.gpu.set_per_process_memory_fraction(0.75)
#tf.config.gpu.set_per_process_memory_growth(True)

# prepare training and testing set


# training
def create_model(input_shape, checkpoint_epoch=0):
    height,width = input_shape
    path = "C:/Users/Boris/Desktop/DC-UNet-main/models/working_models"
    if checkpoint_epoch==0:
        base_model = DCUNet(height=height, width=width, channels=3)
    else:
        # this line is used if we're using checkpoints by epoch
        #base_model = load_model(f"{path}/ep_{checkpoint_epoch:02d}.h5",custom_objects={"focal_tversky":focal_tversky,"dice_coef":dice_coef,"jacard":jacard})
        base_model = DCUNet(height=height, width=width, channels=3)
        base_model.load_weights('results\\weights_checkpoint')
    model = base_model
    return model
#'''
#starting params
PATH = "C:\\Users\\Boris\\Desktop\\DC-UNet-main\\DATASET"
CHECKPOINT_EP = 0
WORKERS = 2
batch_size = 4
EPOCHS = 1
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

df = pd.concat([df_origs],axis=0)#aug_df
x_train, x_test, y_train, y_test = train_test_split(df[0], df[1], test_size=0.2,
                                                      stratify=df[2], random_state=42)

#klase = [f.split(' ')[0] for f in x_train]

klase_1=[]
for orig in x_train:
    if 'benign' in orig:
        klase_1.append('benign')
    elif 'malignant' in orig:
        klase_1.append('malignant')
    elif 'normal' in orig:
        klase_1.append('normal')

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125,
                                                      stratify=klase_1, random_state=42)

valid_predict = ValidPredict(valid_data=(x_val,y_val),img_size=(WIDTH,HEIGHT))
#train_generator = My_Generator(x_train, y_train, PATH, batch_size=batch_size, is_train=True,IMG_size=(WIDTH,HEIGHT)) #accurate order of width and height
train_mixup = My_Generator(x_train, y_train, PATH, batch_size=batch_size, is_train=True, mix=False, augment=False,IMG_size=(WIDTH,HEIGHT))
valid_generator = My_Generator(x_val, y_val, PATH, batch_size=batch_size, is_train=False,IMG_size=(WIDTH,HEIGHT))
save_images_callback = SaveImages(raw_data=(x_train,y_train))
best_save= tf.keras.callbacks.ModelCheckpoint(
    'C:\\Users\\Boris\\Desktop\\DC-UNet-main\\results\\model_best.h5', monitor='val_loss', verbose=1, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch',
    options=None)
weights_save= tf.keras.callbacks.ModelCheckpoint(
    'C:\\Users\\Boris\\Desktop\\DC-UNet-main\\results\\weights.h5', monitor='val_loss', verbose=1, save_best_only=False,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None)
model_checkpoint= tf.keras.callbacks.ModelCheckpoint(
    'C:\\Users\\Boris\\Desktop\\DC-UNet-main\\results\\model_checkpoint.h5', monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None)
csv_logger = CSVLogger(filename='C:\\Users\\Boris\\Desktop\\DC-UNet-main\\results\\v1_training_log.csv', separator=',', append=True)
callbacks_list = [csv_logger,best_save,model_checkpoint,weights_save]#,end_eval ,save_images_callback,valid_predict
# different loss functions
#model = create_model((HEIGHT,WIDTH),checkpoint_epoch=CHECKPOINT_EP) # (height,width)
#model.compile(optimizer=Adam(1e-5), loss=iou_loss, metrics=[tf.losses.categorical_crossentropy,dice_coef, jacard, tversky, 'accuracy'])
model = unet_model(1)
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy',tversky,jacard,dice_coef])
print(len(x_val))
print(int(np.ceil(float(len(x_val)) / float(batch_size))))
print(x_val)
x_val_list,y_val_mask_list = load_files(r"C:\Users\Boris\Desktop\DC-UNet-main",x_val,y_val,(WIDTH,HEIGHT))


model.fit(train_mixup, epochs=EPOCHS,
                          steps_per_epoch=int(np.ceil(float(len(x_train)) / float(batch_size))),
                          validation_data=(x_val_list,y_val_mask_list),
                          callbacks=callbacks_list,verbose=1)


weights = model.get_weights()
#print(weights[0])
model1 = unet_model(1)
model1.load_weights('results\\weights.h5')
weights_1 = model1.get_weights()
#print(np.shape(weights[0][0]))
#print('sledeci')
#print(np.shape(weights_1[0][0]))
#print(len(np.array(weights).reshape([-1])))
#print(np.testing.assert_array_equal(weights,weights))
#tf.keras.utils.plot_model(model, show_shapes=True)
for (layer,layer1) in zip(model.layers,model1.layers):
    result = np.array_equiv(layer.get_weights(),layer1.get_weights())
    #print(layer.get_config())
    #print(layer1.get_config())
    #print(result)
    if result ==False:
        print('Faulty layer')
        print(np.array_equiv(layer.get_weights(), layer1.get_weights()))
        print(f"layer shapes: {np.shape(layer.get_weights())};{np.shape(layer1.get_weights())}")
        print(layer.get_config())
        print(layer1.get_config())
        #print(layer.get_config())
        #print(layer1.get_config())
        #print(np.shape(layer.get_weights()[0]))
        #print(np.shape(layer1.get_weights()[0]))
        result1 = np.array_equiv(layer.get_weights()[0], layer1.get_weights()[0])
        #print(np.sum(np.subtract(layer.get_weights()[0],layer1.get_weights()[0])))
        print(result1)
        result1 = np.array_equiv(layer.get_weights()[1], layer1.get_weights()[1])
        print(result1)
        #print('Previous layer')
        #print(previous_layer)
        #print(previous_layer1)
    previous_layer = layer.get_config()
    previous_layer1 = layer1.get_config()
averagek= []

for x in x_val_list:
    x_val = np.expand_dims(x, axis=0)
    y_pred = model.predict(x_val)
    y_pred1 = model.predict(x_val)
    #print(np.array_equal(y_pred,y_pred1))
    #print(np.sum(np.subtract(y_pred,y_pred1)))
    print(np.allclose(y_pred,y_pred1))

print(np.sum(averagek)/len(averagek))