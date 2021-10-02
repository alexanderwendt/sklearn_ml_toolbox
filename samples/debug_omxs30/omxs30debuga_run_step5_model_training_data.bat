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


echo #=================================================#
echo # Training Model for Temporal Datasets #
echo #=================================================#
python %script_prefix%\step50_train_model_from_pipe.py --config_path=%config_file%
rem train the final model
python %script_prefix%\step50_train_model_from_pipe.py --config_path=%config_file% --config_section="ModelFinal"