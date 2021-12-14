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
set CONFIG=%THISFILENAME:ts4X_hyperparametersearch_=%
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
python %script_prefix%\step42_analyze_training_time.py --config_path=./config/%CONFIG%.ini
python %script_prefix%\step43_wide_hyperparameter_search_all.py --config_path=./config/%CONFIG%.ini --execute_wide=True -debug
python %script_prefix%\step44_narrow_hyperparameter_search_all.py --config_path=./config/%CONFIG%.ini
python %script_prefix%\step45_define_precision_recall.py --config_path=./config/%CONFIG%.ini

echo #=================================================#
echo # Training Model Evaluation for Temporal Datasets #
echo #=================================================#
