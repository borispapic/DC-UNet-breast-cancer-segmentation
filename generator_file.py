import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import csv
import re
import cv2
import string
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from tensorflow.keras.losses import Huber
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse_met
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.utils import shuffle
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
import tensorflow.keras
from tensorflow.python.keras import backend as k
from tensorflow.keras.models import Model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import (ModelCheckpoint,Callback,LearningRateScheduler,EarlyStopping,ReduceLROnPlateau,CSVLogger)

PATH = "C:\\Users\\Boris\\Desktop\\DC-UNet-main\\DATASET"

class My_Generator(Sequence):
    def __init__(self, image_filenames, labels, dest,
                 batch_size, is_train=True,
                 mix=False, augment=False, IMG_size=(128, 96)):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.IMG_size = IMG_size
        self.dest = dest
        self.is_train = is_train
        self.is_augment = augment
        if (self.is_train):
            self.on_epoch_end()
        self.is_mix = mix

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if (self.is_train):
            return self.train_generate(batch_x, batch_y)
        return self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        if (self.is_train):
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
        else:
            pass

    def mix_up(self, x, y):
        lam = np.random.beta(0.2, 0.4)
        ori_index = np.arange(int(len(x)))
        index_array = np.arange(int(len(x)))
        np.random.shuffle(index_array)

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]
        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        return mixed_x, mixed_y

    def train_generate(self, batch_x, batch_y):
        batch_images = []
        batch_masks = []
        for (sample, label) in zip(batch_x, batch_y):
            if str(sample)[0].isdigit():
                path = f"{PATH}/augmented_images/images/{sample}"
            else:
                path = f"{PATH}/Original/{sample}"
            img = cv2.imread(path)
            #img = bens_processing(img, self.IMG_size)
            img = cv2.resize(img, self.IMG_size,interpolation=cv2.INTER_CUBIC)
            #if (self.is_augment):
            #    img = seq_boris.augment_image(img)
            #img = normalization_fn(img)
            batch_images.append(img)
            if str(label)[0].isdigit():
                path = f"{PATH}/augmented_images/labels/{label}"
            else:
                path = f"{PATH}/Ground Truth/{label}"
            mask = cv2.imread(path)
            # img = bens_processing(img, self.IMG_size)
            mask = cv2.resize(mask, self.IMG_size, interpolation=cv2.INTER_CUBIC)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # img = normalization_fn(img)
            batch_masks.append(mask)

        batch_images = np.array(batch_images, np.float32) / 255
        batch_masks = np.array(batch_masks, np.float32) / 255
        batch_masks = np.round(batch_masks, 0)
        if (self.is_mix):
            batch_images, batch_y = self.mix_up(batch_images, batch_masks)
        return batch_images, batch_masks

    def valid_generate(self, batch_x, batch_y):
        batch_images = []
        batch_masks = []
        for (sample, label) in zip(batch_x, batch_y):
            # print(sample)
            if str(sample)[0].isdigit():
                path = f"{PATH}/augmented_images/images/{sample}"
            else:
                path = f"{PATH}/Original/{sample}"
            img = cv2.imread(path)
            #img = bens_processing(img, self.IMG_size)
            img = cv2.resize(img, self.IMG_size,interpolation=cv2.INTER_CUBIC)
            #img = normalization_fn(img)
            batch_images.append(img)
            if str(label)[0].isdigit():
                path = f"{PATH}/augmented_images/labels/{label}"
            else:
                path = f"{PATH}/Ground Truth/{label}"
            mask = cv2.imread(path)
            # img = bens_processing(img, self.IMG_size)
            mask = cv2.resize(mask, self.IMG_size, interpolation=cv2.INTER_CUBIC)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # img = normalization_fn(img)
            batch_masks.append(mask)

        batch_images = np.array(batch_images, np.float32) / 255
        batch_masks = np.array(batch_masks, np.float32) / 255
        batch_masks = np.round(batch_masks, 0)
        return batch_images, batch_masks


class EndEvaluation(Callback):
    def __init__(self, validation_data=(),raw_data = (), batch_size=32, interval=1):
        super(Callback, self).__init__()
        self.x_test,self.y_test = raw_data
        self.interval = interval
        self.batch_size = batch_size
        self.valid_generator, self.y_val = validation_data
        self.history = []

    def on_epoch_begin(self, epoch, logs={}):
        # print(f'starting epoch: {epoch}')
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs={}):
        print(b'b')

