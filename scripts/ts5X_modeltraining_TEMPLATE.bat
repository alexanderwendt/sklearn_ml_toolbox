@echo off

echo #===========================================#
echo # Alexander Wendts Machine Learning Toolbox #
echo #===========================================#

:: define config file to use
set CONFIG="debug_timedata_omxs30_datapreparation"
set script_prefix="..\..\sklearn_ml_toolbox"
set env="sklearn"

::Extract the model name from the current file name
set THISFILENAME=%~n0
set CONFIG=%THISFILENAME:ts5X_modeltraining_=%
echo Load config file %CONFIG%.ini

echo setup environment %env%
call conda activate %env%

echo #===========================================#
echo # Generate Dataset #
echo #===========================================#


echo #===========================================#
echo # Data Analysis and Preprocessing #
echo #===========================================#


echo #===========================================#
echo # Model Training #
echo #===========================================#


echo #=================================================#
echo # Training Model for Temporal Datasets #
echo #=================================================#
python %script_prefix%\step50_train_model_from_pipe.py --config_path=./config/%CONFIG%.ini --config_section="ModelTrain"
rem train the final model
python %script_prefix%\step50_train_model_from_pipe.py --config_path=./config/%CONFIG%.ini --config_section="ModelFinal"