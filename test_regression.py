import subprocess
import os

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return False
    print(f"Success: {command}")
    return True

config_path = "samples/grades/config/grades_regression.ini"

steps = [
    f"python step20_generate_grades.py -conf {config_path}",
    f"python step30_clean_raw_data.py -conf {config_path}",
    # step31 might not be needed if no specific adaptations are required, but let's see
    f"python step31_adapt_features.py -conf {config_path}",
    f"python step35_perform_feature_selection.py -conf {config_path}",
    f"python step36_split_training_validation.py -conf {config_path}",
    f"python step43_wide_hyperparameter_search_all.py -conf {config_path} -debug",
    f"python step44_narrow_hyperparameter_search_all.py -conf {config_path}",
    f"python step50_train_model_from_pipe.py -conf {config_path} -sec Model",
    f"python step60_evaluate_model.py -conf {config_path} -sec EvaluationTraining"
]

for step in steps:
    if not run_command(step):
        break

print("Testing complete.")
