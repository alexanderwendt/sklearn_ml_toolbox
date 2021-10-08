import json

import joblib
import utils.execution_utils as exe

def load_evaluation_data(conf, config_section="Evaluation"):
    '''


    '''

    X_path = conf[config_section].get('features_in')
    y_path = conf[config_section].get('outcomes_in')
    labels_path = conf[config_section].get('labels_in')
    model_in = conf[config_section].get('model_in')
    ext_param_in = conf[config_section].get('ext_param_in')

    # Load X and y
    X_val, _, y_val = exe.load_data(X_path, y_path)

    # Labels
    labels = exe.load_labels(labels_path)

    # Load model
    model = joblib.load(model_in)
    print("Loaded trained evaluation model from ", model_in)
    print("Model", model)

    # Load external parameters
    with open(ext_param_in, 'r') as fp:
        external_params = json.load(fp)

    return X_val, y_val, labels, model, external_params