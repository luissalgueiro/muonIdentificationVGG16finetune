import pandas as pd
import pytz, datetime
import pytz
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os, fnmatch
from astropy.io import fits
import cv2
import sys


Path='/home/gpu/Documentos/CONNIEData/eventsFits/'
li=sys.argv
fileName=li[1]
Data = pd.read_csv(fileName)

muestras=20
window_height=200
j=0
os.system('echo missing fits >fileFits ')
for row in Data.iterrows():
    ohdu=row[1].ohdu
    n=row[1].events
    runID=row[1].runID
    j=j+1
    np.random.seed(j*n)
    if(j==25):
        j=0
   # print(row[1])
    selected=np.sort(np.random.randint(low=1,high=n, size=(muestras)))
    flag=True
    while(flag):
        u, c = np.unique(selected, return_counts=True)
        dup = u[c > 1]
        if(len(dup>0)):
            result = np.where(selected == dup[0])
            selected[result[0][0]]=np.random.randint(low=1,high=n, size=(1))
        else:
            flag=False
 
    for i in selected:
        fitsName=Path+str(runID)+'/catalog' +str(runID)+ '_' +str(ohdu)+'_' + str(i) + '.fits'
        if  not(os.path.isfile(fitsName)):
            os.system('echo '+fitsName+' >>fileFits ') 
        pngName='assetsN/'+str(runID)+ '_' +str(ohdu)+'_' + str(i) +'.png'
       # if  os.path.isfile(pngName):
        #    continue
       # else:
       #     print(pngName)
        #    os.system('echo '+pngName+' >>filePng ') 
        img = fits.getdata(fitsName)
        
        aspect_ratio = float(img.shape[1])/float(img.shape[0])
        window_width = window_height/aspect_ratio
        res = cv2.resize(np.sqrt(img), dsize=(int(window_height),int(window_width)), interpolation=cv2.INTER_NEAREST)
          #subplot del evento
        fig = plt.figure(figsize=(60,20))
        ax0 = fig.add_subplot(121)
        plt.imshow(res,cmap='hot_r')
        plt.text(10, -40,str(runID)+' '+str(ohdu)+' '+str(i), fontsize=fz)
        plt.text(10, -10,str(img.shape[1])+'x'+str(img.shape[0]), fontsize=fz)

        plt.colorbar()
        plt.rc('xtick', labelsize=35) 
        plt.rc('ytick', labelsize=35) 
        ax0 = plt.gca()
        ax0.invert_yaxis()
        ax0.tick_params(labelsize=35)
        ax0.tick_params(labelsize=35) 

                #subplot del histograma
        ax1 = fig.add_subplot(122)
        #plt.hist(img, bins = 60)
        sns.distplot(img);
        plt.yscale('log', nonposy='clip')
        plt.rc('xtick', labelsize=35) 
        plt.rc('ytick', labelsize=35) 
        ax1 = plt.gca()
        fig = plt.gcf()
        plt.savefig(pngName)
        plt.close(fig)
        