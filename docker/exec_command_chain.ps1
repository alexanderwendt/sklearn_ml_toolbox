#docker exec `<container> bash -c "command1 ; command2 ; command3"

#docker exec e8a2eb5e4580 bash -c "cd /sklearn-ml-toolbox/projects/debug_omxs30 && chmod 777 *.sh && ./is7X_prediction_debug_timedata_omxs30_infer_svm.sh"

docker run `
-v C:\Projekte\23_Machine_Learning_Toolbox\sklearn_ml_toolbox_samples\debug_omxs30:/sklearn-ml-toolbox/projects/debug_omxs30 `
-p 80:80 `
sklearn_ml_toolbox:py38 `
/bin/bash `
-c "cd /sklearn-ml-toolbox/projects/debug_omxs30 && chmod 777 *.sh && ./is7X_prediction_debug_timedata_omxs30_infer_svm.sh"