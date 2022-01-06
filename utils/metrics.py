import json

from sklearn.metrics import make_scorer, precision_score, recall_score, accuracy_score, f1_score
import warnings

class Metrics:
    def __init__(self, config):
        self.refit_scorer_name = config['Training'].get('refit_scorer_name')

        # Load custom scorer setup (fallback = 0)
        average_method = config['Training'].get('average_method', fallback='macro')
        used_labels = config['Training'].get('labels', fallback=None)
        if used_labels=='None':
            used_labels = None
        else:
            used_labels = json.loads(used_labels)
        pos_label = config['Training'].get('pos_label', fallback=1)
        pos_label = json.loads(pos_label)
        if not average_method=='average':
            pos_label=1
        self.scorers = self.__generate_scorers(average_method, used_labels, pos_label)

    def __generate_scorers(self, average_method, labels, pos_label):
        #Average method ‘micro’, ‘macro’, ‘samples’,’weighted’, ‘binary’} or None, default=’binary’
        average_method = average_method #'macro'  # Calculate Precision1...Precisionn and Recall1...Recalln separately and average.
        #The set of labels to include when average != 'binary', and their order if average is None. Labels present in
        # the data can be excluded, for example to calculate a multiclass average ignoring a majority negative class,
        # while labels not present in the data will result in 0 components in a macro average. For multilabel targets,
        # labels are column indices. By default, all labels in y_true and y_pred are used in sorted order.
        used_labels = labels #None
        #The class to report if average='binary' and the data is binary. If the data are multiclass or multilabel,
        # this will be ignored; setting labels=[pos_label] and average != 'binary' will report scores for that label
        # only.
        pos_label = pos_label #1
        # It is good to increase the weight of smaller classes

        warnings.warn("Precision has option zero_division=0 instead of warn")

        scorers = {
            'precision_score': make_scorer(precision_score, zero_division=0,
                                           labels=used_labels, pos_label=pos_label, average=average_method),
            'recall_score': make_scorer(recall_score, labels=used_labels, pos_label=pos_label, average=average_method),
            'accuracy_score': make_scorer(accuracy_score),
            'f1_score': make_scorer(f1_score, labels=used_labels, pos_label=pos_label, average=average_method)
        }

        return scorers