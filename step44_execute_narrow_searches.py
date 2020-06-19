import argparse
import pickle

from IPython.core.display import display
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC

import step40functions as step40
import DataSupportFunctions as sup
import Sklearn_model_utils as modelutil
import numpy as np
import copy
import DatavisualizationFunctions as vis

## %% First run with a wide grid search
# Minimal set of parameter to test different grid searches
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, Normalizer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from pickle import dump



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step 4.3 - Execute wide grid search for SVM')
    parser.add_argument("-exe", '--execute_wide', default=False,
                        help='Execute Training', required=False)
    parser.add_argument("-d", '--data_path', default="04_Model/prepared_input.pickle",
                        help='Prepared data', required=False)

    args = parser.parse_args()

    #if not args.pb and not args.xml:
    #    sys.exit("Please pass either a frozen pb or IR xml/bin model")

    # Execute wide search
    #execute_wide_run(execute_search=args.execute_wide, data_input_path=args.data_path)

    print("=== Program end ===")