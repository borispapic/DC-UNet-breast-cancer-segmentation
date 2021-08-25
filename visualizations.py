import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Za training
# Epoch/Accuracy/Dice/Jaccard/tversky Loss
# Za log
# Epoha/Jaccard/Dice/Tversky


def plot_function(PATH):
    metric = 'Jaccard'
    #tr_log = pd.read_csv(f"{PATH}\\v1_training_log.csv")
    #val_log = pd.read_csv(f"{PATH}\\log.txt",delimiter=',',header=None)
    df= pd.read_csv(f"{PATH}\\v1_training_log.csv")
    #print(df.iloc[:,:4].head(10))
    #plt.plot(np.arange(1,301),tr_metrics.iloc[:,1])
    line1, = plt.plot(np.arange(1,201),df.iloc[:200,4],'b',label='Focal Tversky training loss')
    line2, = plt.plot(np.arange(1,201),df.iloc[:200:,9],'orange',label='Focal Tversky validation loss')
    plt.legend(handles=[line1, line2])
    plt.grid()
    plt.xlabel('Epoch(s)')
    plt.ylabel('Loss')
    #plt.ylim(0,1)
    plt.show()

def get_confusion_matrix(y_true,y_pred):
    # Not really needed in this case
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
    print(f'True positives: {tp}, False positives: {fp}, False negatives: {fn}, True negatives: {tn}')

PATH = "C:\\Users\\Boris\\Desktop\\DC-UNet-main" +'\\results' #+ "\\models\\working_models\\MODEL V2\\"
PATH = "C:\\Users\\Boris\\Desktop\\DC-UNet-main\\models\\working_models\\" +"MODEL V19 - focal tversky AUGMENTATION no aug in test data"
plot_function(PATH)