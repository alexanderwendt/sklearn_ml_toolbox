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
python %script_prefix%\step20_generate_groundtruth_stockmarket.py --config_path=%config_file%
python %script_prefix%\step21_generate_features.py --config_path=%config_file% -debug
python %script_prefix%\step22_adapt_dimensions.py --config_path=%config_file%

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
python %script_prefix%\step42_analyze_training_time_svm.py --config_path=%config_file%
python %script_prefix%\step43_wide_hyperparameter_search_svm.py --config_path=%config_file% --execute_wide=True -debug
python %script_prefix%\step44_narrow_hyperparameter_search_svm.py --config_path=%config_file%
python %script_prefix%\step45_define_precision_recall.py --config_path=%config_file%

echo #=================================================#
echo # Training Model for Temporal Datasets #
echo #=================================================#
python %script_prefix%\step50_train_model_from_pipe.py --config_path=%config_file%
rem train the final model
python %script_prefix%\step50_train_model_from_pipe.py --config_path=%config_file% --config_section="ModelFinal"

echo #=================================================#
echo # Model Evaluation for Temporal Datasets #
echo #=================================================#
python %script_prefix%\step60_evaluate_model.py --config_path=%config_file% --config_section="EvaluationTraining"
python %script_prefix%\step61_evaluate_model_temporal_data.py --config_path=%config_file% --config_section="EvaluationTraining"

python %script_prefix%\step60_evaluate_model.py --config_path=%config_file%
python %script_prefix%\step61_evaluate_model_temporal_data.py --config_path=%config_file%



