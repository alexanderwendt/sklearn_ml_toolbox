echo #===========================================#
echo # Alexander Wendts Machine Learning Toolbox #
echo #===========================================#

rem define config file to use
set config_file="config/debug_timedata_omxS30_inference.ini"
set script_prefix="..\.."
set env="sklearn"

echo setup environment
call conda activate %env%

echo #===========================================#
echo # Generate Dataset #
echo #===========================================#
rem python step20_generate_groundtruth_stockmarket.py --config_path=%config_file%
python %script_prefix%\step21_generate_features.py --config_path=%config_file% -debug
python %script_prefix%\step22_adapt_dimensions.py --config_path=%config_file%

echo #===========================================#
echo # Data Analysis and Preprocessing #
echo #===========================================#


echo #===========================================#
echo # Model Training #
echo #===========================================#


echo #=================================================#
echo # Training Model Evaluation for Temporal Datasets #
echo #=================================================#

