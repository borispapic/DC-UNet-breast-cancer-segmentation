import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PATH = "C:\\Users\\Boris\\Desktop\\DC-UNet-main"

tr_log = pd.read_csv(f"{PATH}\\results\\v1_training_log.csv")
val_log = pd.read_csv(f"{PATH}\\models\\log.txt",delimiter=',',header=None)
print(val_log.iloc[:,:4].head(10))
#plt.plot(np.arange(1,301),tr_metrics.iloc[:,1])
line1, = plt.plot(np.arange(1,301),tr_log.iloc[:,2],label='Jaccard')
line2, = plt.plot(np.arange(1,301),tr_log.iloc[:,3],label='Dice')
plt.legend(handles=[line1, line2])
plt.xlabel('Epoch(s)')
plt.ylabel('Jaccard index')
plt.show()