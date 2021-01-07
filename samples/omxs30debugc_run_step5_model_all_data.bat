echo #===========================================#
echo # Alexander Wendts Machine Learning Toolbox #
echo #===========================================#

rem define config file to use
set config_file="config/debug_timedata_omxs30_complete_model.ini"

echo setup environment
call conda activate env_ml

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
echo # Training Model Evaluation for Temporal Datasets #
echo #=================================================#
python ..\step50_train_model_from_pipe.py --config_path=%config_file%
