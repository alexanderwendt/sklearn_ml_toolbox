echo #===========================================#
echo # Alexander Wendts Machine Learning Toolbox #
echo #===========================================#

rem define config file to use
set config_file="config/debug_timedata_omxS30.ini"

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
python step42_analyze_training_time_svm.py --config_path=%config_file%
python step43_wide_hyperparameter_search_svm.py --config_path=%config_file% --execute_wide=True -debug
python step44_narrow_hyperparameter_search_svm.py --config_path=%config_file%
python step45_define_precision_recall.py --config_path=%config_file%

echo #=================================================#
echo # Training Model Evaluation for Temporal Datasets #
echo #=================================================#
