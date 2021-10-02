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
python %script_prefix%\step30_clean_raw_data.py --config_path=%config_file%
python %script_prefix%\step31_adapt_features.py --config_path=%config_file%
rem python %script_prefix%\step32_search_hyperparameters.py --config_path=%config_file%
python %script_prefix%\step33_analyze_data.py --config_path=%config_file%
python %script_prefix%\step34_analyze_temporal_data.py --config_path=%config_file%
python %script_prefix%\step35_perform_feature_selection.py --config_path=%config_file%
python %script_prefix%\step36_split_training_validation.py --config_path=%config_file%

echo #===========================================#
echo # Model Training #
echo #===========================================#


echo #=================================================#
echo # Training Model Evaluation for Temporal Datasets #
echo #=================================================#

