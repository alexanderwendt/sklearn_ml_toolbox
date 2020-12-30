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
rem python step20_generate_groundtruth_stockmarket.py --config_path=%config_file%
rem python step21_generate_features.py --config_path=%config_file%
rem python step22_dimension_adapter.py --config_path=%config_file%

echo #===========================================#
echo # Data Analysis and Preprocessing #
echo #===========================================#
python step31_clean_raw_data.py --config_path=%config_file%
python step32_adapt_features.py --config_path=%config_file%
python step33_hyperparameter_search.py --config_path=%config_file%
python step34_data_analysis.py --config_path=%config_file%
python step35_data_analysis_temporal.py --config_path=%config_file%
python step36_feature_selection.py --config_path=%config_file%

echo #===========================================#
echo # Model Training #
echo #===========================================#