import pickle #Save data
import step40functions as step40

from sklearn.svm import SVC
import matplotlib.pyplot as plt

# %% Load input
model_input_path = "04_Model" + "/" + "prepared_input.pickle"
f = open(model_input_path,"rb")
prepared_data = pickle.load(f)
print("Loaded data: ", prepared_data)
X_train = prepared_data['X_train']
y_train = prepared_data['y_train']
X_test = prepared_data['X_test']
y_test = prepared_data['y_test']
y_classes = prepared_data['y_classes']
scorers = prepared_data['scorers']
refit_scorer_name = prepared_data['refit_scorer_name']

# %% test
scorer = scorers[refit_scorer_name]

# %% Baseline test
baseline_results = step40.execute_baseline_classifier(X_train, y_train, X_test, y_test, y_classes, scorer)
print("Baseline results=", baseline_results)

# %% Estimate training duration
run_training_estimation = True

if run_training_estimation==True:
    #Set test range
    test_range = list(range(100, 6500+1, 500))
    #test_range = list(range(100, 1000, 200))
    print("Test range", test_range)

    # SVM model
    # Define the model
    model_clf = SVC()
    xaxis, durations, scores = step40.estimate_training_duration(model_clf, X_train, y_train, X_test, y_test, test_range, scorer)

    # Paint figure
    plt.figure()
    plt.plot(xaxis, durations)
    plt.xlabel('Number of training examples')
    plt.ylabel('Duration [s]')
    plt.title("Training Duration")
    plt.show()

    plt.figure()
    plt.plot(xaxis, scores)
    plt.xlabel('Number of training examples')
    plt.ylabel('F1-Score on cross validation set (=the rest). Size={}'.format(X_test.shape[0]))
    plt.title("F1 Score Improvement With More Data")
    plt.show()

print("=== Script end ===")