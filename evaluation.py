from model import DCUNet
from tensorflow.keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import os
import sys
from tensorflow.keras import backend as K

def make_list(PATH,augmentations=False):


    original = os.listdir(f'{PATH}\\DATASET\\Original')
    del_ext = [f.split('.')[0] for f in original]
    mask = [f + '_mask.png' for f in del_ext]
    klase = [f.split(' ')[0] for f in original]
    df = pd.concat([pd.Series(original), pd.Series(mask), pd.Series(klase)], axis=1)
    if augmentations:
        original_aug = os.listdir(f'{PATH}\\DATASET\\augmented_images\\images')
        del_ext_mask = [f.split('.')[0] for f in original_aug]
        mask_aug = [f + '_mask.png' for f in del_ext_mask]
        aug_klase = class_vector(original_aug)
        aug_df = pd.concat([pd.Series(original_aug), pd.Series(mask_aug), pd.Series(aug_klase)], axis=1)
        df = pd.concat([df,aug_df], axis=0)
    return df

def class_vector(x):
    klase_1=[]
    for orig in x:
        if 'benign' in orig:
            klase_1.append('benign')
        elif 'malignant' in orig:
            klase_1.append('malignant')
        elif 'normal' in orig:
            klase_1.append('normal')
    return klase_1

def load_files(PATH,df_x,df_y,IMG_size=()):
    mask_list =[]
    original_list=[]

    for i in range(df_x.shape[0]):
        sample = df_x.iloc[i]
        if str(sample)[0].isdigit():
            path = f"{PATH}\\DATASET\\augmented_images\\images\\{sample}"
        else:
            path = f"{PATH}\\DATASET\\Original\\{sample}"
        img = cv2.imread(f"{path}")
        img = cv2.resize(img, IMG_size, interpolation=cv2.INTER_CUBIC)
        original_list.append(img)
        label = df_y.iloc[i]
        if str(label)[0].isdigit():
            path = f"{PATH}\\DATASET\\augmented_images\\labels\\{label}"
        else:
            path = f"{PATH}\\DATASET\\Ground Truth\\{label}"
        mask = cv2.imread(path)
        mask = cv2.resize(mask, IMG_size, interpolation=cv2.INTER_CUBIC)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_list.append(mask)
    original_list = np.array(original_list, np.float32) / 255
    mask_list = np.array(mask_list, np.float32) / 255
    mask_list = np.round(mask_list, 0)
    return original_list,mask_list

def create_model(input_shape):
    height,width = input_shape
    base_model = DCUNet(height=height, width=width, channels=3)
    model = base_model
    return model

def load_model(PATH,HEIGHT,WIDTH):
    #model = create_model((HEIGHT, WIDTH))
    #model.load_weights(f'{PATH}\\weights_best')
    model = tf.keras.models.load_model('my_model.h5',custom_objects={"iou_loss":iou_loss,"dice_coef":dice_coef,"jacard":jacard,"tversky":tversky})
    return model

def show_images(PATH,HEIGHT,WIDTH,epoch):
    df = make_list(PATH)
    df = df.sample(20)
    original_list,mask_list = load_files(PATH,df,(WIDTH,HEIGHT)) #jebeni cv2
    model = load_model(PATH,HEIGHT,WIDTH)
    try:
        os.makedirs(f'results\\images\\epoch_{epoch+1:02d}')
    except:
        pass
    for i in range(len(original_list)):
        X = np.expand_dims(original_list[i], axis=0)
        yp = model.predict(x=X, verbose=1)
        yp = np.round(yp, 0)
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(original_list[i])
        plt.title('Input')
        plt.subplot(1, 3, 2)
        plt.imshow(mask_list[i].reshape(mask_list[i].shape[0], mask_list[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1, 3, 3)
        plt.imshow(yp[0].reshape(yp[0].shape[0], yp[0].shape[1]))
        plt.title('Prediction')
        plt.show()
        plt.savefig(f'results\\images\\epoch_{epoch + 1:02d}\\i' + str(i) + '.png', format='png')
        plt.close()


def dice_coef(y_true, y_pred):
    smooth = 1.0  #0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = tf.math.round(y_pred_f)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = tf.math.round(y_pred_f)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)
    res = intersection/union
    return res


def dice_coef_loss(y_true,y_pred):
    smooth = 1.0  # 0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_l= (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice_l

def iou_loss(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)
    return 1 - intersection/union

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    y_pred_pos = tf.round(y_pred_pos)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.75
    smooth = 1
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.75
    smooth = 1
    res = (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    return 1 - res

def focal_tversky(y_true,y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.75
    smooth = 1
    pt_1 = (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def parse_image(img,label):
    text = img.numpy().decode('utf-8')
    #text = img
    if str(text)[0].isdigit():
        path = f"DATASET/augmented_images/images/{text}"
    else:
        path = f"DATASET/Original/{text}"
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img,channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [96, 128])

    text = label.numpy().decode('utf-8')
    #text = label
    if str(text)[0].isdigit():
        path = f"DATASET/augmented_images/labels/{text}"
    else:
        path = f"DATASET/Ground Truth/{text}"
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask,channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.image.resize(mask, [96, 128])
    #img,mask = normalize(img,mask)
    return img,mask

def set_shapes(image, label):
    image,label = tf.py_function(parse_image, [image,label], [tf.float32,tf.float32])
    image.set_shape((96, 128, 3))
    label.set_shape((96, 128, 1))
    return image, label

def create_model(input_shape, checkpoint_epoch=0):
    height,width = input_shape
    #path = "C:/Users/Boris/Desktop/DC-UNet-main/models/working_models"
    if checkpoint_epoch==0:
        base_model = DCUNet(height=height, width=width, channels=3)
    else:
        # this line is used if we're using checkpoints by epoch
        #base_model = load_model(f"{path}/ep_{checkpoint_epoch:02d}.h5",custom_objects={"focal_tversky":focal_tversky,"dice_coef":dice_coef,"jacard":jacard})
        base_model = DCUNet(height=height, width=width, channels=3)
        base_model.load_weights('results\\weights.h5')
    model = base_model
    return model
PATH = "C:\\Users\\Boris\\Desktop\\DC-UNet-main"
HEIGHT = 96
WIDTH = 128
#show_images(PATH,HEIGHT,WIDTH,epoch=1) #
