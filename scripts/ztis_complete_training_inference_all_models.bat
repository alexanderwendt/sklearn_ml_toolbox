@echo off
set CONFIG_TRAIN_DATAPREPARATION=debug_timedata_omxs30_datapreparation_train
set CONFIG_INFER_DATAPREPARATION=debug_timedata_omxs30_infer_svm

set CONFIG_TRAIN_SVM=debug_timedata_omxs30_train_svm
set CONFIG_TRAIN_XGBOOST=debug_timedata_omxs30_train_xgboost
set CONFIG_TRAIN_SVM_LINEAR=debug_timedata_omxs30_train_svm_linear
set CONFIG_TRAIN_KNN=debug_timedata_omxs30_train_knn
set CONFIG_TRAIN_GNB=debug_timedata_omxs30_train_gnb

set CONFIG_INFER_SVM=debug_timedata_omxs30_infer_svm
set CONFIG_INFER_XGBOOST=debug_timedata_omxs30_infer_xgboost
set CONFIG_INFER_SVM_LINEAR=debug_timedata_omxs30_infer_svm_linear
set CONFIG_INFER_KNN=debug_timedata_omxs30_infer_knn
set CONFIG_INFER_GNB=debug_timedata_omxs30_infer_gnb


echo ##################################################################
echo # Data Preparation Training                                      #
echo ##################################################################

call ts23X_complete_datapreparation_%CONFIG_TRAIN_DATAPREPARATION%.bat



echo ##################################################################
echo # Training and Evaluation Model                                  #
echo ##################################################################

call ts4567X_complete_training_%CONFIG_TRAIN_SVM%.bat
call ts4567X_complete_training_%CONFIG_TRAIN_XGBOOST%.bat
call ts4567X_complete_training_%CONFIG_TRAIN_SVM_LINEAR%.bat
call ts4567X_complete_training_%CONFIG_TRAIN_KNN%.bat
call ts4567X_complete_training_%CONFIG_TRAIN_GNB%.bat



echo ##################################################################
echo # Data Preparation Inference                                     #
echo ##################################################################

call is23X_complete_datapreparation_%CONFIG_INFER_DATAPREPARATION%.bat



echo ##################################################################
echo # Inference SVM                                                  #
echo ##################################################################

call is7X_prediction_%CONFIG_INFER_SVM%.bat
call is7X_prediction_%CONFIG_INFER_XGBOOST%.bat
call is7X_prediction_%CONFIG_INFER_SVM_LINEAR%.bat
call is7X_prediction_%CONFIG_INFER_KNN%.bat
call is7X_prediction_%CONFIG_INFER_GNB%.bat



echo Training and Inference finished.
