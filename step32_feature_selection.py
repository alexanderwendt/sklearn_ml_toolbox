





print("Skip feature analysis=", skip_feature_analysis)
print("Skip feature selection", skip_feature_selection)


# Generating filenames for saving the files
features_filename = target_directory + "/" + conf['dataset_name'] + "_features" + ".csv"
model_features_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_features_for_model" + ".csv"
outcomes_filename = target_directory + "/" + conf['dataset_name'] + "_outcomes" + ".csv"
model_outcomes_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_outcomes_for_model" + ".csv"
labels_filename = target_directory + "/" + conf['dataset_name'] + "_labels" + ".csv"
source_filename = target_directory + "/" + conf['dataset_name'] + "_source" + ".csv"
#Modified labels
model_labels_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_labels_for_model" + ".csv"
#Columns for feature selection
selected_feature_columns_filename = target_directory + "/" + conf['dataset_name'] + "_" + conf['class_name'] + "_selected_feature_columns.csv"

print("=== Paths ===")
print("Input Features: ", features_filename)
print("Output Features: ", model_features_filename)
print("Input Outcomes: ", outcomes_filename)
print("Output Outcomes: ", model_outcomes_filename)
print("Labels: ", labels_filename)
print("Original source: ", source_filename)
print("Labels for the model: ", model_labels_filename)
print("Selected feature columns: ", selected_feature_columns_filename)