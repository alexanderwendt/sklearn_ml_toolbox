echo #===========================================#
echo # Alexander Wendts Machine Learning Toolbox #
echo #===========================================#

rem define config file to use
set config_file="config/debug_timedata_omxS30.ini"
set script_prefix="..\.."
set env="sklearn"

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
python %script_prefix%\step42_analyze_training_time_svm.py --config_path=%config_file%
python %script_prefix%\step43_wide_hyperparameter_search_svm.py --config_path=%config_file% --execute_wide=True -debug
python %script_prefix%\step44_narrow_hyperparameter_search_svm.py --config_path=%config_file%
python %script_prefix%\step45_define_precision_recall.py --config_path=%config_file%

echo #=================================================#
echo # Training Model Evaluation for Temporal Datasets #
echo #=================================================#
