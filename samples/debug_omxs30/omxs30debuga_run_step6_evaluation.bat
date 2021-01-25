echo #===========================================#
echo # Alexander Wendts Machine Learning Toolbox #
echo #===========================================#

rem define config file to use
set config_file="config/debug_timedata_omxS30.ini"
set script_prefix="..\.."
set env="scikit"

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
python %script_prefix%\step60_evaluate_model.py --config_path=%config_file%
python %script_prefix%\step61_evaluate_model_temporal_data.py --config_path=%config_file%

echo #=================================================#
echo # Prediction #
echo #=================================================#
