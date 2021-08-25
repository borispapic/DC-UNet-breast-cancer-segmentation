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
from evaluation import load_files,load_model,make_list,class_vector,parse_image,set_shapes,create_model
from second_model import unet_model
print(sys.version)
print(tf.__version__)
from os import walk
#tf.config.list_physical_devices('GPU')
#tf.config.gpu.set_per_process_memory_fraction(0.75)
#tf.config.gpu.set_per_process_memory_growth(True)

# prepare training and testing set


# training


#'''
#starting params
PATH = "C:\\Users\\Boris\\Desktop\\DC-UNet-main\\DATASET"
tf.random.set_seed(42)
np.random.seed(42)
CHECKPOINT_EP = 31
AUGMENTATIONS = False
BATCH_SIZE = 8#64
BUFFER_SIZE = 1000
LR = 1e-3
OPTIMIZER = Adam(LR,clipnorm=0.001)
EPOCHS = 200
HEIGHT = 96 #96
WIDTH = 128 #128
LOSS= tversky_loss
X = []
Y = []

original = os.listdir(f'{PATH}/Original')
del_ext = [f.split('.')[0] for f in original]
mask = [f + '_mask.png' for f in del_ext]
klase = [f.split(' ')[0] for f in original]

original_aug = os.listdir(f'{PATH}\\augmented_images\\images')
del_ext_mask = [f.split('.')[0] for f in original_aug]
mask_aug = [f + '_mask.png' for f in del_ext_mask]
aug_klase = class_vector(original_aug)
aug_df = pd.concat([pd.Series(original_aug), pd.Series(mask_aug), pd.Series(aug_klase)], axis=1)

df = pd.concat([pd.Series(original),pd.Series(mask),pd.Series(klase)],axis=1)
if AUGMENTATIONS:
    df = pd.concat([df,aug_df],axis=0)#aug_df

x_train, x_test, y_train, y_test = train_test_split(df[0], df[1], test_size=0.2,
                                                      stratify=df[2], random_state=42)

#klase = [f.split(' ')[0] for f in x_train]

klase_1=class_vector(x_train)
x_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
g = pd.concat([x_train,y_train,pd.Series(klase_1)],axis=1)
g.columns =[0,1,2]
x_train, x_val, y_train, y_val = train_test_split(g[0], g[1], test_size=0.125,
                                                      stratify=g[2], random_state=42)

valid_predict = ValidPredict(valid_data=(x_val,y_val),img_size=(WIDTH,HEIGHT))
#train_generator = My_Generator(x_train, y_train, PATH, batch_size=batch_size, is_train=True,IMG_size=(WIDTH,HEIGHT)) #accurate order of width and height
#train_mixup = My_Generator(x_train, y_train, PATH, batch_size=BATCH_SIZE, is_train=True, mix=False, augment=False,IMG_size=(WIDTH,HEIGHT))
#valid_generator = My_Generator(x_val, y_val, PATH, batch_size=BATCH_SIZE, is_train=False,IMG_size=(WIDTH,HEIGHT))
save_images_callback = SaveImages(raw_data=(x_train,y_train))
best_save= tf.keras.callbacks.ModelCheckpoint(
    'C:\\Users\\Boris\\Desktop\\DC-UNet-main\\results\\model_best.h5', monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch',
    options=None)
weights_save= tf.keras.callbacks.ModelCheckpoint(
    'C:\\Users\\Boris\\Desktop\\DC-UNet-main\\results\\weights.h5', monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None)
model_checkpoint= tf.keras.callbacks.ModelCheckpoint(
    'C:\\Users\\Boris\\Desktop\\DC-UNet-main\\results\\model_checkpoint.h5', monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=True, mode='auto', save_freq='epoch',
    options=None)
csv_logger = CSVLogger(filename='C:\\Users\\Boris\\Desktop\\DC-UNet-main\\results\\v1_training_log.csv', separator=',', append=True)
callbacks_list = [csv_logger,best_save,model_checkpoint,weights_save,save_images_callback,valid_predict]#,end_eval ,save_images_callback,valid_predict
## GETTING TRAIN GEN
ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#dataset = ds.map(lambda x,y: parse_image(x,y),num_parallel_calls=tf.data.AUTOTUNE)
dataset = ds.map(lambda x,y: set_shapes(x,y), num_parallel_calls=tf.data.AUTOTUNE)
train = dataset.cache().prefetch(10).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
## GETTING VAL GEN
ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
dataset = ds.map(lambda x,y: set_shapes(x,y), num_parallel_calls=tf.data.AUTOTUNE)
val = dataset.cache().prefetch(10).batch(BATCH_SIZE).repeat()#shuffle izmedju

model = create_model((HEIGHT,WIDTH),checkpoint_epoch=CHECKPOINT_EP) # (height,width)
model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, jacard, tversky, 'accuracy'])

#x_val_list,y_val_mask_list = load_files(r"C:\Users\Boris\Desktop\DC-UNet-main",x_val,y_val,(WIDTH,HEIGHT))
model.fit(
    train,
    validation_data=val,
    steps_per_epoch=int(np.ceil(float(len(x_train)) / float(BATCH_SIZE))),
    validation_steps = int(np.ceil(float(len(x_val)) / float(BATCH_SIZE))),
    epochs=EPOCHS,
    verbose=1,initial_epoch=CHECKPOINT_EP,callbacks=callbacks_list)

## CREATING TEST
ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
dataset = ds.map(lambda x,y: set_shapes(x,y), num_parallel_calls=tf.data.AUTOTUNE)
test = dataset.cache().prefetch(10).batch(BATCH_SIZE).repeat()#shuffle izmedju
loss,result_dice,result_jaccard,result_tversky,accuracy = model.evaluate(test,steps=int(np.ceil(float(len(x_val)) / float(BATCH_SIZE))))
print("Test results on trained model without saving:")
print(loss,result_dice,result_jaccard,result_tversky,accuracy)
model_saved = create_model((HEIGHT,WIDTH),checkpoint_epoch=CHECKPOINT_EP)
model_saved.load_weights('results/model_best.h5')
model_saved.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[dice_coef, jacard, tversky, 'accuracy'])
loss,result_dice,result_jaccard,result_tversky,accuracy =model_saved.evaluate(test,steps=int(np.ceil(float(len(x_val)) / float(BATCH_SIZE))))
print("Test results on trained model after saving:")
print(loss,result_dice,result_jaccard,result_tversky,accuracy)

loss,result_dice,result_jaccard,result_tversky,accuracy =model_saved.evaluate(val,steps=int(np.ceil(float(len(x_val)) / float(BATCH_SIZE))))
print("Validation results on trained model after saving:")
print(loss,result_dice,result_jaccard,result_tversky,accuracy)