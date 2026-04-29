import json

from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
import warnings

class Metrics:
    def __init__(self, config, labels=None):
        self.refit_scorer_name = config['Training'].get('refit_scorer_name')
        self.problem_type = config['Common'].get('problem_type', fallback='classification')

        if self.problem_type == 'classification':
            # Load custom scorer setup (fallback = 0)
            average_method = config['Training'].get('average_method', fallback='macro')
            used_labels = labels
            pos_label = None
            if not average_method == 'average':
                pos_label = 1
            self.scorers = self.__generate_classification_scorers(average_method, used_labels, pos_label)
        elif self.problem_type == 'regression':
            self.scorers = self.__generate_regression_scorers()
        else:
            raise ValueError(f"Unknown problem_type: {self.problem_type}")

    def __generate_classification_scorers(self, average_method, labels, pos_label):
        from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
        import warnings
        # Average method ‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’
        # It is good to increase the weight of smaller classes
        warnings.warn("Precision has option zero_division=0 instead of warn")

        scorers = {
            'precision_score': make_scorer(precision_score, zero_division=0,
                                           labels=labels, pos_label=pos_label, average=average_method),
            'recall_score': make_scorer(recall_score, labels=labels, pos_label=pos_label, average=average_method),
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, labels=labels, pos_label=pos_label, average=average_method)
        }

        return scorers

    def __generate_regression_scorers(self):
        from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
        scorers = {
            'mse': make_scorer(mean_squared_error, greater_is_better=False),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'r2': make_scorer(r2_score)
        }
        return scorers