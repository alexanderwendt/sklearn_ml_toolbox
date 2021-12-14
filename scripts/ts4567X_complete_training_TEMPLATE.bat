@echo off

:: define config file to use
::set CONFIG="debug_timedata_omxs30_datapreparation"
set script_prefix="..\..\sklearn_ml_toolbox"
set env="sklearn"

::Extract the model name from the current file name
set THISFILENAME=%~n0
set CONFIG=%THISFILENAME:ts4567X_complete_training_=%
echo Load config file %CONFIG%.ini

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
python %script_prefix%\step42_analyze_training_time.py --config_path=./config/%CONFIG%.ini
::python %script_prefix%\step43_wide_hyperparameter_search_all.py --config_path=./config/%CONFIG%.ini --execute_wide=True -debug
python %script_prefix%\step43_wide_hyperparameter_search_all.py --config_path=./config/%CONFIG%.ini --execute_wide=True
python %script_prefix%\step44_narrow_hyperparameter_search_all.py --config_path=./config/%CONFIG%.ini
python %script_prefix%\step45_define_precision_recall.py --config_path=./config/%CONFIG%.ini

echo #=================================================#
echo # Training Model for Temporal Datasets #
echo #=================================================#
python %script_prefix%\step50_train_model_from_pipe.py --config_path=./config/%CONFIG%.ini --config_section="ModelTrain"
rem train the final model
python %script_prefix%\step50_train_model_from_pipe.py --config_path=./config/%CONFIG%.ini --config_section="ModelFinal"

echo #=================================================#
echo # Model Evaluation for Temporal Datasets #
echo #=================================================#
python %script_prefix%\step60_evaluate_model.py --config_path=./config/%CONFIG%.ini --config_section="EvaluationTraining"
python %script_prefix%\step61_evaluate_model_temporal_data.py --config_path=./config/%CONFIG%.ini --config_section="EvaluationTraining"

python %script_prefix%\step60_evaluate_model.py --config_path=./config/%CONFIG%.ini --config_section="EvaluationValidation"
python %script_prefix%\step61_evaluate_model_temporal_data.py --config_path=./config/%CONFIG%.ini --config_section="EvaluationValidation"


echo #=================================================#
echo # Post Processing #
echo #=================================================#
python %script_prefix%\step71_value_postprocessing.py --config_path=./config/%CONFIG%.ini --config_section="EvaluationValidation"
python %script_prefix%\step72_backtesting.py --config_path=./config/%CONFIG%.ini --config_section="EvaluationValidation"


echo Training and Inference finished.
