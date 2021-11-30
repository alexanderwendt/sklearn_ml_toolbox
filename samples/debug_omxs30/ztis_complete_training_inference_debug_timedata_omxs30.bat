@echo off
set CONFIG_TRAIN_DATAPREPARATION=debug_timedata_omxs30_datapreparation_train
set CONFIG_TRAIN_SVM=debug_timedata_omxs30_train_svm
set CONFIG_TRAIN_XGBOOST=debug_timedata_omxs30_train_xgboost
set CONFIG_INFER_DATAPREPARATION=debug_timedata_omxs30_infer_svm
set CONFIG_INFER_SVM=debug_timedata_omxs30_infer_svm
set CONFIG_INFER_XGBOOST=debug_timedata_omxs30_infer_xgboost


echo ##################################################################
echo # Data Preparation Training                                      #
echo ##################################################################
echo Use config: %CONFIG_TRAIN_DATAPREPARATION%

call ts2X_generation_%CONFIG_TRAIN_DATAPREPARATION%.bat
call ts3X_datapreparation_%CONFIG_TRAIN_DATAPREPARATION%.bat


echo ##################################################################
echo # Training and Evaluation SVM                                    #
echo ##################################################################
echo Use config: %CONFIG_TRAIN_SVM%

call ts4X_hyperparametersearch_%CONFIG_TRAIN_SVM%.bat
call ts5X_modeltraining_%CONFIG_TRAIN_SVM%.bat
call ts6X_evaluation_%CONFIG_TRAIN_SVM%.bat


echo ##################################################################
echo # Training and Evaluation XGBoost                                #
echo ##################################################################
echo Use config: %CONFIG_TRAIN_XGBOOST%

call ts4X_hyperparametersearch_%CONFIG_TRAIN_XGBOOST%.bat
call ts5X_modeltraining_%CONFIG_TRAIN_XGBOOST%.bat
call ts6X_evaluation_%CONFIG_TRAIN_XGBOOST%.bat


echo ##################################################################
echo # Data Preparation Inference                                     #
echo ##################################################################
echo Use config: %CONFIG_INFER_DATAPREPARATION%

call is2X_generation_%CONFIG_INFER_DATAPREPARATION%.bat
call is3X_datapreparation_%CONFIG_INFER_DATAPREPARATION%.bat

echo ##################################################################
echo # Inference SVM                                                  #
echo ##################################################################
echo Use config: %CONFIG_INFER_SVM%

call is7X_prediction_%CONFIG_INFER_SVM%.bat

echo ##################################################################
echo # Inference XGBoost                                              #
echo ##################################################################
echo Use config: %CONFIG_INFER_XGBOOST%

call is7X_prediction_%CONFIG_INFER_XGBOOST%.bat

echo Training and Inference finished.
