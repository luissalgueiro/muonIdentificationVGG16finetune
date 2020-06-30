In this repository present the scripts and notebooks were we develop muon identifier Convolutional Neural Network for CONNIE 1x5 experiment. 
Since all the runs are in root files we deloped severel extraction scripts:

Then 4000 events were randomly sample:

A human-determined categorical labels were assined to our train-test sample:


Then we apply data augmentation to balance our categories and make the model approximately invariant to random noise.


Next, we summarize the model after training and validation process applied with KFold crossvalidation.


Finally, we compute the classification results, the confusion matrix and the performance of the model.