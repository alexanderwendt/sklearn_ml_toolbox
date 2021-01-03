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
rem python step31_clean_raw_data.py --config_path=%config_file%
rem python step32_adapt_features.py --config_path=%config_file%
rem python step33_hyperparameter_search.py --config_path=%config_file%
rem python step34_data_analysis.py --config_path=%config_file%
rem python step35_data_analysis_temporal.py --config_path=%config_file%
rem python step36_feature_selection.py --config_path=%config_file%

echo #===========================================#
echo # Model Training #
echo #===========================================#
rem python step41_prepare_input.py --config_path=%config_file%
rem python step42_training_predictions.py --data_path="config/paths.pickle"
rem python step43_execute_wide_search.py --data_path="config/paths.pickle" --execute_wide -debug
rem python step43_execute_wide_search.py --data_path="config/paths.pickle" -debug
rem python step44_execute_narrow_searches.py --data_path="config/paths.pickle"
rem python step45_define_precision_recall.py --data_path="config/paths.pickle"
rem python step46_train_evaluate_model.py --data_path="config/paths.pickle"
rem python step47_train_final_model.py --data_path="config/paths.pickle"

echo #=================================================#
echo # Training Model Evaluation for Temporal Datasets #
echo #=================================================#
python step50_evaluate_model_temporal_data.py --data_path="config/paths.pickle"
