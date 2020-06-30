import numpy as np
import ROOT #ROOT para Python
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import random
from sklearn import preprocessing
from astropy.io import fits
import sys
from os import listdir
import os.path
import root_pandas as rpd
#li=sys.argv
#print(li)
#file=li[1]
#ohdus=li[2]
#Path=li[3]
#PathOut=li[4]
Path='/home/gpu/Documentos/CONNIEData/'
PathOut='/home/gpu/Documentos/CONNIEData/events'
files=['hpix_scn_osi_raw_gain_catalog_data_6031_to_6230_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_6322_to_6521_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_6522_to_6721_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_6722_to_6921_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_6922_to_7121_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_7122_to_7321_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_7322_to_7521_v3.0.root']

ohdus=['2','3','4','5']
for file in files:
    print('File '+file)
#    cut='flag==0 && hPixFlag==0 && xMin>130 && xMax<3960 && yMin>130 && yMax<880'
    cut='flag==0 && hPixFlag==0 && xMin>10 && xMax<4103 && yMin>100 && yMax<890'
    ddata= rpd.read_root([Path+file], columns=['runID','flag','hPixFlag','expoStart','ohdu','xMax','xMin','yMax',\
    'yMin','xPix','yPix','ePix'],where=cut,key='hitSumm')
    for ohdu in ohdus:
        dataFil= ddata[ddata['ohdu'] == int(ohdu)]
        print(dataFil.shape[0])
        runIDs=np.unique(dataFil.runID.values)
        print(len(runIDs))
        for runID in runIDs:
            print('runid'+ str(runID))
            fitsCatName =PathOut+'CatFits/catalog' +str(runID)+ '_' +str(ohdu)+'.fits'
           
            dataF = dataFil[dataFil['runID'] == runID]
            print(dataF.shape[0])
         
            Ai = sparse.coo_matrix((4113, 900))
            for event_number in  range(1,dataF.shape[0]):
                try:
                    X = dataF.xPix.values[event_number]
                    Y = dataF.yPix.values[event_number]
                    W = dataF.ePix.values[event_number]

                    #outlier
                    q1 = np.quantile(W,0.25)
                    q3 = np.quantile(W,0.75)
                    liminf = q1 - (1.5*(q3-q1))

                    flag = W<liminf
                    W[flag]=0
                    if(max(Y)>900):
                        print('AlertaYmax')
                    #matriz sparse del evento
                    img = sparse.coo_matrix((W+min(W),(Y-min(Y),X-min(X))),shape = (max(Y)+1-min(Y),max(X)+1-min(X)))
                    Ai=Ai+sparse.coo_matrix((W, (X,Y)),shape=(4113, 900))
                    fitsName = PathOut+'Fits/' +str(runID)+ '/catalog' +str(runID)+ '_' +str(ohdu)+'_' + str(event_number) + '.fits'
                    fits.writeto(fitsName,img.toarray(),overwrite=True)
                except ValueError:
                    print('error en catalog ' + file + str(event_number))
            fits.writeto(fitsCatName,Ai.toarray())

