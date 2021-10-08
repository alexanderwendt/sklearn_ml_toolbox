from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
import warnings

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