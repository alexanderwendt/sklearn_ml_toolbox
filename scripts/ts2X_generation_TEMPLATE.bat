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
set CONFIG=%THISFILENAME:ts2X_generation_=%
echo Load config file %CONFIG%.ini

echo setup environment %env%
call conda activate %env%



echo #===========================================#
echo # Generate Dataset #
echo #===========================================#
python %script_prefix%\step20_generate_groundtruth_stockmarket.py --config_path=./config/%CONFIG%.ini
python %script_prefix%\step21_generate_features.py --config_path=./config/%CONFIG%.ini -debug
python %script_prefix%\step22_adapt_dimensions.py --config_path=./config/%CONFIG%.ini

echo #===========================================#
echo # Data Analysis and Preprocessing #
echo #===========================================#


echo #===========================================#
echo # Model Training #
echo #===========================================#


echo #=================================================#
echo # Training Model Evaluation for Temporal Datasets #
echo #=================================================#