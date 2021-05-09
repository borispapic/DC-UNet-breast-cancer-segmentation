import os
import shutil
from os import listdir
from os.path import isfile, join
from pathlib import Path

TO_PATH= "C:\\Users\\Boris\\Desktop\\DC-UNet-main\\DATASET"
FROM_PATH = "C:\\Users\\Boris\\Desktop\\DC-UNet-main\\RAW"

def move_function(FROM_PATH,TO_PATH):
    for path, subdirs, files in os.walk(FROM_PATH):
        for file in files:
            if file.endswith("_mask.png"):
                subFolder = os.path.join(TO_PATH+"\\Ground Truth\\", file)
                print("MASK "+subFolder)
                shutil.copy(os.path.join(path,file), subFolder)
            elif file.endswith("_mask_1.png") or file.endswith("_mask_2.png"):
                print(f"passed: {file}")
            else:
                subFolder = os.path.join(TO_PATH+"\\Original\\",file)
                print("ORIGINAL " + subFolder)
                shutil.copy(os.path.join(path,file), subFolder)

def checking_function(FROM_PATH,TO_PATH):
    all_masks = []
    for file_2 in os.listdir(TO_PATH+"\\Ground Truth"):
        all_masks.append(file_2)
    #print(all_masks)
    for file in os.listdir(TO_PATH+"\\Original"):
        original_name = file[:-4] + "_mask.png"
        #print(original_name)# retrieves the last 5 characters in the filename.
        if original_name not in all_masks:
            print(f"{original_name} doesn't have an equivalent")

move_function(FROM_PATH,TO_PATH)
checking_function(FROM_PATH,TO_PATH)