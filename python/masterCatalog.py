import pandas as pd
import numpy as np
from scipy import stats
import os, fnmatch
import root_pandas as rpd
import pytz, datetime

files=['hpix_scn_osi_raw_gain_catalog_data_6031_to_6230_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_6322_to_6521_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_6522_to_6721_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_6722_to_6921_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_6922_to_7121_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_7122_to_7321_v3.0.root',
'hpix_scn_osi_raw_gain_catalog_data_7322_to_7521_v3.0.root']
ohdus=[2,3,4,5,6,7,8,9,10,11,12,13,14,15]
Path='/home/gpu/Documentos/CONNIEData/'
PathOut='/home/gpu/Documentos/diego/Connie/Mayodatos/events'
Nevents= pd.DataFrame({'runID': [],'ohdu' : [],'expoStart' :[],'events' : [],'eventscut' : []})

for file in files:
    print('File '+file)
    cut='flag==0 && hPixFlag==0 && xMin>10 && xMax<4103 && yMin>100 && yMax<890'
    #cut='ohdu < 6 && flag==0 && hPixFlag==0 && xMin>130 && xMax<3960 && yMin>130 && yMax<880'
    #cut=' flag==0 && hPixFlag==0 && xMin>130 && xMax<3960 && yMin>130 && yMax<880'

    dataFil = rpd.read_root([Path+file], columns=['runID','expoStart','ohdu','xMax','xMin','yMax','yMin'],where=cut,key='hitSumm')
    for ohdu in ohdus:
        print('Ohdu '+str(ohdu))
        print(dataFil.shape[0])
        dataF = dataFil[dataFil['ohdu'] == ohdu]
        runIDs=np.unique(dataF.runID.values)
        print(len(runIDs))
        expoStart=np.unique(dataF.expoStart.values)
        print(len(expoStart))
        flag=np.logical_and(dataF.xMin>130, dataF.xMax<3960)
        flag=np.logical_and(flag, dataF.yMin>130)
        flag=np.logical_and(flag,dataF.yMax<880)
        dataF['flag']=flag
        events= pd.DataFrame({'runID': runIDs ,'ohdu' : ohdu*np.ones_like(runIDs),'expoStart' : expoStart,'events' : dataF.groupby('runID').size().values,'eventscut' : dataF.groupby('runID')['flag'].sum().values})

        Nevents= pd.concat([Nevents,events])
Nevents = Nevents.astype(({'runID': int,'ohdu' : int,'expoStart' :int,'events' : int}))
Nevents['datetime']=pd.to_datetime(Nevents.expoStart, unit='s',utc=True)
Nevents.to_csv('NewmasterCatalogALLOHDU.csv',index=False)