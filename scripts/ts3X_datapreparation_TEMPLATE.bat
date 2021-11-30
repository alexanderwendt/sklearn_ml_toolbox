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
set CONFIG=%THISFILENAME:ts3X_datapreparation_=%
echo Load config file %CONFIG%.ini

echo setup environment %env%
call conda activate %env%

echo #===========================================#
echo # Generate Dataset #
echo #===========================================#


echo #===========================================#
echo # Data Analysis and Preprocessing #
echo #===========================================#
python %script_prefix%\step30_clean_raw_data.py --config_path=./config/%CONFIG%.ini
python %script_prefix%\step31_adapt_features.py --config_path=./config/%CONFIG%.ini
rem python %script_prefix%\step32_search_hyperparameters.py --config_path=./config/%CONFIG%.ini
python %script_prefix%\step33_analyze_data.py --config_path=./config/%CONFIG%.ini
python %script_prefix%\step34_analyze_temporal_data.py --config_path=./config/%CONFIG%.ini
python %script_prefix%\step35_perform_feature_selection.py --config_path=./config/%CONFIG%.ini
python %script_prefix%\step36_split_training_validation.py --config_path=./config/%CONFIG%.ini

echo #===========================================#
echo # Model Training #
echo #===========================================#


echo #=================================================#
echo # Training Model Evaluation for Temporal Datasets #
echo #=================================================#

