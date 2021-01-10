echo #===========================================#
echo # Alexander Wendts Machine Learning Toolbox #
echo #===========================================#

rem define config file to use
set config_file="config/debug_timedata_omxS30.ini"

echo setup environment
call conda activate scikit

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
python ..\step60_evaluate_model.py --config_path=%config_file%
python ..\step61_evaluate_model_temporal_data.py --config_path=%config_file%

echo #=================================================#
echo # Prediction #
echo #=================================================#
