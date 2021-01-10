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
python ..\step20_generate_groundtruth_stockmarket.py --config_path=%config_file%
python ..\step21_generate_features.py --config_path=%config_file% -debug
python ..\step22_adapt_dimensions.py --config_path=%config_file%

echo #===========================================#
echo # Data Analysis and Preprocessing #
echo #===========================================#
python ..\step30_clean_raw_data.py --config_path=%config_file%
python ..\step31_adapt_features.py --config_path=%config_file%
rem python ..\step32_search_hyperparameters.py --config_path=%config_file%
python ..\step33_analyze_data.py --config_path=%config_file%
python ..\step34_analyze_temporal_data.py --config_path=%config_file%
python ..\step35_perform_feature_selection.py --config_path=%config_file%
python ..\step36_split_training_validation.py --config_path=%config_file%

echo #===========================================#
echo # Model Training #
echo #===========================================#
python ..\step42_analyze_training_time_svm.py --config_path=%config_file%
python ..\step43_wide_hyperparameter_search_svm.py --config_path=%config_file% --execute_wide=True -debug
python ..\step44_narrow_hyperparameter_search_svm.py --config_path=%config_file%
python ..\step45_define_precision_recall.py --config_path=%config_file%

echo #=================================================#
echo # Training Model for Temporal Datasets #
echo #=================================================#
python ..\step50_train_model_from_pipe.py --config_path=%config_file%

echo #=================================================#
echo # Model Evaluation for Temporal Datasets #
echo #=================================================#
python ..\step60_evaluate_model.py --config_path=%config_file%
python ..\step61_evaluate_model_temporal_data.py --config_path=%config_file%



