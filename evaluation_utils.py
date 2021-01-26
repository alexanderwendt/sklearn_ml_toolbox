import json
import warnings

import joblib
from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
import execution_utils as exe

class Metrics:
    def __init__(self, config):
        self.refit_scorer_name = config['Training'].get('refit_scorer_name')
        self.scorers = self.__generate_scorers()

    def __generate_scorers(self):
        average_method = 'macro'  # Calculate Precision1...Precisionn and Recall1...Recalln separately and average.
        # It is good to increase the weight of smaller classes

        warnings.warn("Precision has option zero_division=0 instead of warn")
        
        scorers = {
            'precision_score': make_scorer(precision_score, average=average_method, zero_division=0),
            'recall_score': make_scorer(recall_score, average=average_method),
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, average=average_method)
        }

        return scorers

def load_evaluation_data(conf):
    '''


    '''

    X_path = conf['Evaluation'].get('features_in')
    y_path = conf['Evaluation'].get('outcomes_in')
    labels_path = conf['Evaluation'].get('labels_in')
    model_in = conf['Evaluation'].get('model_in')
    ext_param_in = conf['Evaluation'].get('ext_param_in')

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