import pandas as pd
import os

def renaming_files(PATH):
    PATH = 'C:\\Users\\Boris\\Desktop\\DC-UNet-main\\DATASET\\labels'
    files = os.listdir(PATH)
    for index, file in enumerate(files):
        print(file)
        os.rename(os.path.join(PATH, file), os.path.join(PATH, file[:-9]+'.png'))

def renaming_augmented_files():
    PATH = 'C:\\Users\\Boris\\Desktop\\DC-UNet-main\\DATASET\\augmented_images\\labels'
    files = os.listdir(PATH)
    for index, file in enumerate(files):
        print(file)
        os.rename(os.path.join(PATH, file), os.path.join(PATH, file[:-4]+'_mask.png'))

renaming_augmented_files()