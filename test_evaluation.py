
from evaluation import load_files,load_model,make_list,class_vector
from evaluation import tversky_loss,tversky,focal_tversky,jacard,dice_coef,iou_loss,dice_coef_loss
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import sys
from numpy.random import seed
seed(42)# keras seed fixing import tensorflow as tf
tf.random.set_seed(42)# tensorflow seed fixing
PATH = "C:\\Users\\Boris\\Desktop\\DC-UNet-main"
model_path = r'C:\Users\Boris\Desktop\DC-UNet-main\results'
data_path = r'C:\Users\Boris\Desktop\DC-UNet-main\results'
HEIGHT = 96
WIDTH = 128
df = make_list(PATH,augmentations=False)
#df = df.sample(20)
model = load_model(model_path,HEIGHT,WIDTH)
x_train, x_test, y_train, y_test = train_test_split(df[0], df[1], test_size=0.2,
                                                      stratify=df[2], random_state=42)
klase_1 = class_vector(x_train)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125,
                                                      stratify=klase_1, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
original_list,mask_list = load_files(PATH,x_test,y_test,(WIDTH,HEIGHT)) #jebeni cv2 #change here

g=0
average_tversky = []
average_dice = []
average_jaccard = []
for i in range(len(original_list)):
    X = np.expand_dims(original_list[i], axis=0)
    y_pred = model.predict(x=X, verbose=1)
    y_pred = np.round(y_pred)
    #print(mask_list[i].shape)
    #print(np.squeeze(y_pred).shape)
    y_pred = np.squeeze(y_pred)
    #sys.exit()
    result_tversky = tversky(mask_list[i],y_pred)
    result_jaccard = jacard(mask_list[i],y_pred)
    result_dice = dice_coef(mask_list[i], y_pred)
    print(f"Dice coef: {result_dice}; Result jaccard: {result_jaccard}; Result tversky: {result_tversky}")
    average_tversky.append(result_tversky.numpy())
    average_jaccard.append(result_jaccard.numpy())
    average_dice.append(result_dice.numpy())
    #plt.figure(figsize=(20, 10))
    #plt.subplot(1, 3, 1)
    #plt.imshow(original_list[i])
    #plt.title('Input')
    #plt.subplot(1, 3, 2)
    #plt.imshow(mask_list[i].reshape(mask_list[i].shape[0], mask_list[i].shape[1]))
    #plt.title('Ground Truth')
    #plt.subplot(1, 3, 3)
    #plt.imshow(y_pred[0].reshape(y_pred[0].shape[0], y_pred[0].shape[1]))
    #plt.title(f'Prediction, {result_tversky.numpy()}')
    #plt.savefig(f'results\\train_images\\' + str(i) + '.png', format='png')
    #g +=1
    #if (g % 20)==0:
        #plt.show()
    plt.close()


#average_tversky,average_jaccard,average_dice = dataset_solver(original_list,mask_list)
klase_2 = class_vector(x_test)# change here
average_jaccard=pd.Series(average_jaccard).fillna(1)
new_df = pd.concat([pd.Series(klase_2),average_jaccard,pd.Series(average_tversky),pd.Series(average_dice)],axis=1)
new_df.columns = ['class','jaccard','tversky','dice']
print(new_df)
#new_df.to_csv(f'{model_path}\\results_test.csv',columns=['class','jaccard','tversky','dice'])

df= pd.read_csv(f"{data_path}\\v1_training_log.csv",index_col=0)
best_column = df[df['val_loss']==df['val_loss'].min()]
print(best_column)
print(f"Average tversky on test set is {np.mean(average_tversky)}; Tversky on validation: {best_column['val_tversky'].to_numpy()[0]}; "
      f"Tversky on training: {best_column['tversky'].to_numpy()[0]}")
print(f"Average jaccard on test set is {np.mean(average_jaccard)}; Jaccard on validation: {best_column['val_jacard'].to_numpy()[0]}; "
      f"Jaccard on training: {best_column['jacard'].to_numpy()[0]}")
print(f"Average dice on test set is {np.mean(average_dice)}; Dice on validation: {best_column['val_dice_coef'].to_numpy()[0]}; "
      f"Dice on training: {best_column['dice_coef'].to_numpy()[0]}")


print(new_df.groupby(['class']).mean())
print(new_df.groupby(['class']).count())

