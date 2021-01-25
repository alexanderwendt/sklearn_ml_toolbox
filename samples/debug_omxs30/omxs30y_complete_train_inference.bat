echo ##################################################################
echo # Test Training and Analysis, Training whole Model and Inference #
echo ##################################################################

echo ##################################################################
echo # Start Training with Training Data                              #
echo ##################################################################

call omxs30debugx_run_step100_complete_training.bat



echo ##################################################################
echo # Start Training of the Complete Model                           #
echo ##################################################################

call omxs30debugx_run_step100_train_final_model.bat


echo ##################################################################
echo # Inference                                                      #
echo ##################################################################

call omxs30debugx_run_step100_complete_inference.bat

