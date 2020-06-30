
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time 
import tensorflow as tf
import warnings; warnings.simplefilter('ignore')
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import random
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model,model_from_json
from sklearn.metrics import classification_report, confusion_matrix
from astropy.io import fits
import cv2
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import itertools
from pickle import dump,load

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#my_devices = tf.config.experimental.list_physical_devices(device_type='XLA_GPU')
#tf.config.experimental.set_visible_devices(devices= my_devices, device_type='XLA_GPU')
modelpath='modelos/VGG16-8020crossval/'
json_file = open(modelpath+'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(modelpath+"best_model.h5")
print("Loaded model from disk")


Data = pd.read_csv('NewmasterCatalogALLOHDU.csv')

Data.head()

Data.info()





Data.set_index(['datetime'],drop=True, inplace=True)



ohdus=[2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for ohdu in ohdus:
        dataF = Data[Data['ohdu'] == ohdu]
        fig, axes = plt.subplots(1,1, figsize=(20,2), sharex=True)
        dataF['eventscut'].plot(subplots=True,marker='.', markersize=2,color='red', linestyle='None', ax=axes)
        axes.set_ylabel('Events OHDU'+str(ohdu))
        axes.set_xlabel('Date')
        plt.tight_layout()

fig.savefig("EventosAllOHDU.png")



size=112
def preprocessing(rundID, ohdu,i,size):
    
    fits_image_filename =Pwd+str(runID)+'/catalog' +str(runID)+ '_' +str(ohdu)+'_' + str(i) + '.fits'
    try:
         hdul=fits.open(fits_image_filename,memmap=False) 
    except:
        os.system('echo ' +str(runID)+ ',' +str(ohdu)+',' + str(i)+'>>failed.csv')
    matrix = hdul[0].data
    del hdul
    
    #************************
    #funcion que recibe una matriz imagen, y hace el preprocesamiento necesario, jutno con el resize y agregado de 0s
    #*************
    #convertir a numpy array
    data = np.array(matrix)
    #quitar los valores negativos
    data = np.where(data<0, 0, data)
    #convertir a escala logaritmica
   # data = np.uint8(np.log1p(data))
    data = (np.sqrt(data))
    
    #normalizar a 0-255
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    
    #resize
    if data.shape[1] == data.shape[0]:
        result = cv2.resize(data,(size,size), interpolation = cv2.INTER_NEAREST)
        
    elif data.shape[0] > data.shape[1]:
        resized = cv2.resize(data,(int(data.shape[1]*size/data.shape[0]),size), interpolation = cv2.INTER_NEAREST)
        sumar = int((size - resized.shape[1])/2)
        if resized.shape[1]%2 == 0:
            result = np.pad(resized, ((0,0),(sumar,sumar)) , mode='median')
        else:
            result = np.pad(resized, ((0,0),(sumar+1,sumar)) , mode='median')
            
    elif data.shape[1]> data.shape[0]:
        resized = cv2.resize(data,(size,int(data.shape[0]*size/data.shape[1])), interpolation = cv2.INTER_NEAREST)
        sumar = int((size - resized.shape[0])/2)
        if resized.shape[0]%2 == 0:
            result = np.pad(resized, ((sumar,sumar),(0,0)) , mode='median')
        else:
            result = np.pad(resized, ((sumar+1,sumar),(0,0)) , mode='median')
    return result



def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))





Pwd='/tf/Documentos/CONNIEData/eventsFits/'


from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()





os.system('echo runID,ohdu,e_id>failed.csv')
    
muonsCountt=[]
i=0
for ck in chunker(Data,500):
    i=i+1
    muonsCount=[]
    ck.head()
    for row in ck.iterrows():
        #events=row[1].events
        #lista=np.linspace(1, events, events, endpoint=True).astype(int)
        runID=row[1].runID.astype(int)
        ohdu=row[1].ohdu.astype(int)
        maxev=row[1].events.astype(int)
        lista=np.linspace(1, maxev, maxev, endpoint=True).astype(int)
        muon=0
        if(True):
            data=[]
            data = Parallel(n_jobs=num_cores)(delayed(preprocessing)(runID, ohdu,i,size) for i in lista)
            #print(np.shape(data))
            DData =np.expand_dims(data, axis=3)
            DataN = np.concatenate( [DData,DData,DData], axis=-1 )
            BS=200#np.int((len(lista)/10.))
            if(len(lista)<100):
                BS=40
            predIdxsV =loaded_model.predict(DataN, batch_size=BS)
            predIdxs = np.argmax(predIdxsV, axis=1)
           # print(np.sum(predIdxs==1))                     
            muon=muon+np.sum(predIdxs==1)
        print(str(runID)+ ',' +str(ohdu)+ ','+str(len(lista))+',' +str(muon))

        muonsCount.append(muon)
    ck['muonCount']=muonsCountt
    
    muonsCountt=muonsCountt.append(muonsCount)
dump(muonsCountt, open('New80croosmuonCountALLOHDU.pkl', 'wb'))

Data['muonCount']=muonsCountt
Data.to_csv('New80crosscatalogmuonsALLOHDU.csv')



