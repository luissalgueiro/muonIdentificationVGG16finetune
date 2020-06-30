In this repository present, the scripts and notebooks were we develop muon identifier Convolutional Neural Network for CONNIE 1x5 experiment. 
Since all the runs are in root files we deloped several extraction scripts:
-"extracRoot2Fits\masterCatalog.py" script generates the masterCatalog and apply some border cuts and bad events flagged.


- "extracRoot2Fits\GenImgFits.py" extracts individual fits and frame for each OHDU
Then 4000 events were randomly sample:

Human-determined categorical labels were assigned to our train-test sample:


Then we apply data augmentation to balance our categories and make the model approximately invariant to random noise.


Next, we summarize the model after the training and validation process applied with KFold cross-validation.


Finally, we compute the classification results, the confusion matrix, and the performance of the model.