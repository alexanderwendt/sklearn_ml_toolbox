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
set CONFIG=%THISFILENAME:ts6X_evaluation_=%
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
echo # Hyperparameter Search #
echo #===========================================#


echo #=================================================#
echo # Model Training#
echo #=================================================#


echo #=================================================#
echo # Model Evaluation for Temporal Datasets #
echo #=================================================#
python %script_prefix%\step60_evaluate_model.py --config_path=./config/%CONFIG%.ini --config_section="EvaluationTraining"
python %script_prefix%\step61_evaluate_model_temporal_data.py --config_path=./config/%CONFIG%.ini --config_section="EvaluationTraining"

python %script_prefix%\step60_evaluate_model.py --config_path=./config/%CONFIG%.ini --config_section="EvaluationValidation"
python %script_prefix%\step61_evaluate_model_temporal_data.py --config_path=./config/%CONFIG%.ini --config_section="EvaluationValidation"

echo #=================================================#
echo # Prediction #
echo #=================================================#
