@echo off

echo #===========================================#
echo # Alexander Wendts Machine Learning Toolbox #
echo #===========================================#

:: define config file to use
set CONFIG="debug_timedata_omxs30_datapreparation_x"
set script_prefix="..\.."
set env="sklearn"

::Extract the model name from the current file name
set THISFILENAME=%~n0
set CONFIG=%THISFILENAME:is7X_prediction_=%
echo Load config file %CONFIG%.ini

echo setup environment %env%
call conda activate %env%



echo #===========================================#
echo # Generate Dataset #
echo #===========================================#


echo #===========================================#
echo # Data Analysis and Preprocessing #
echo #===========================================#


echo #=================================================#
echo # Model Training#
echo #=================================================#


echo #=================================================#
echo # Model Evaluation for Temporal Datasets #
echo #=================================================#


echo #=================================================#
echo # Prediction #
echo #=================================================#
python %script_prefix%\step70_predict_temporal_data.py --config_path=./config/%CONFIG%.ini --config_section="EvaluationInference"