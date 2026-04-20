# The SKLearn Machine Learning Toolbox

The SKLearn Machine Learning Toolbox is a comprehensive toolchain for data preparation, analysis, model training, and prediction for tabular data. It provides a structured workflow to handle various machine learning tasks, demonstrated with a stock market prediction example.

The toolbox is designed as a template that can be adapted to different datasets and problems. The scripts are organized into numbered steps to guide the user through the machine learning pipeline.

## Project Structure

It is recommended to follow a structured directory layout for your projects to work seamlessly with the toolbox.

-   `sklearn_ml_toolbox/`: The root directory of this toolbox containing the Python scripts.
-   `your_project_name/`: Your project directory.
    -   `annotations/`: Ground truth or label files.
    -   `config/`: Project-specific configuration files (`.ini`).
    -   `data_raw/`: Raw input data.
    -   `data_prepared/`: Processed and cleaned data ready for training.
    -   `models/`: Saved, trained models.
    -   `results/`: Output from the pipeline, such as evaluations, predictions, and plots.
    -   `run_scripts/`: Batch or shell scripts to execute the pipeline steps.

## Setup

1.  **Create a Conda Environment:**
    ```bash
    conda create -n sklearn python=3.12
    conda activate sklearn
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    If you encounter issues, you may need to install some libraries manually from `requirements.txt`.

## Machine Learning Pipeline Workflow

The pipeline is divided into sequential steps, each corresponding to a Python script.

### Step 2X: Data and Feature Generation

This step focuses on generating ground truth labels and features from the raw data.

-   **`step20_generate_groundtruth_stockmarket.py`**: Automatically generates labels from the data. In the example, it identifies long-term trends in stock market data.
-   **`step20_generate_groundtruth_stockmarket_from_annotation.py`**: Loads manually created labels from a CSV file in the `annotations` directory.
-   **`step21_generate_features.py`**: Generates a full set of features from the raw data. For the stock market example, this includes various technical indicators.
-   **`step21_generate_features_reduced_lt.py`**: Generates a reduced, specific set of features tailored for a particular task, like long-term trend prediction.
-   **`step22_adapt_dimensions.py`**: Aligns the dimensions of feature and label dataframes, which can become mismatched after generation steps that use future or past data (e.g., moving averages).

### Step 3X: Data Preparation, Analysis, and Feature Selection

This step cleans the data, performs exploratory data analysis, selects the most relevant features, and splits the data for training.

-   **`step30_clean_raw_data.py`**: Cleans the dataset by handling missing values, correcting data types, and standardizing column names.
-   **`step31_adapt_features.py`**: Prepares features for machine learning models, including tasks like one-hot encoding for categorical variables.
-   **`step32_search_hyperparameters.py`**: A utility script to find optimal hyperparameters for visualization and analysis tools like T-SNE.
-   **`step33_analyze_data.py`**: Performs extensive exploratory data analysis (EDA) with various visualizations like correlation matrices, scatter plots, pair plots, and dimensionality reduction plots (PCA, T-SNE, UMAP) to understand the data structure.
-   **`step34_analyze_temporal_data.py`**: Provides tools for analyzing time-series data, such as autocorrelation plots.
-   **`step35_perform_feature_selection.py`**: Uses various techniques (Lasso, Tree-based, Backward/Recursive Elimination) to identify and select the most significant features for the model.
-   **`step36_split_training_validation.py`**: Splits the dataset into training and validation sets for model training and evaluation.

### Step 4X: Hyperparameter Tuning

This step is for finding the optimal hyperparameters for the machine learning model.

-   **`step42_analyze_training_time.py`**: Establishes a baseline performance by evaluating dummy classifiers and analyzes the training time and performance scaling with respect to the number of samples.
-   **`step43_wide_hyperparameter_search_all.py`**: Performs a broad search across a wide range of discrete hyperparameters (e.g., scalers, samplers, kernels, feature sets) to find the best pipeline components.
-   **`step44_narrow_hyperparameter_search_all.py`**: Conducts a more focused, iterative search on a narrower range of continuous hyperparameters (e.g., C, gamma for SVM) to fine-tune the model.
-   **`step45_define_precision_recall.py`**: Optimizes the decision threshold for classification models by analyzing the precision-recall curve, which is crucial for imbalanced datasets.

### Step 5X: Model Training

-   **`step50_train_model_from_pipe.py`**: Trains the final machine learning model using the complete, optimized pipeline and hyperparameters found in the previous steps. It can train a model on the training set for validation and another on the entire dataset for inference.

### Step 6X: Model Evaluation

This step evaluates the performance of the trained model.

-   **`step60_evaluate_model.py`**: Evaluates the model on the validation set using various metrics and visualizations like confusion matrices and decision boundary plots.
-   **`step61_evaluate_model_temporal_data.py`**: Provides specific evaluation for time-series data, plotting predictions against ground truth over time to visually assess performance.

### Step 7X: Prediction and Post-processing

This final step uses the trained model to make predictions on new, unseen data.

-   **`step70_predict_temporal_data.py`**: Loads a trained model and uses it to predict outcomes for inference data, visualizing the results for temporal data.
-   **`step71_value_postprocessing.py`**: Applies post-processing logic to the raw model predictions. This can include smoothing techniques or rule-based filters to improve the usability of the output.
-   **`step72_backtesting.py`**: Performs a historical simulation to evaluate the financial viability of a trading strategy based on the model's predictions. This is a critical step for any financial machine learning model.

## Configuration

The pipeline is controlled by `.ini` configuration files located in the `config` directory of your project. These files allow you to specify paths, model parameters, and other settings for each step of the process.

## How to Run

The `scripts` and `data_template_folder_structure` directories contain template batch (`.bat`) and shell (`.sh`) scripts for running the pipeline. You can copy and adapt these scripts to your project to execute the steps in the correct order.

## Future Work

-   Integrate more classification and regression models (e.g., Random Forest, KNN, MLP).
-   Expand metrics and visualizations for regression problems.

## Issues

For any issues or suggestions, please open an issue on the GitHub repository or contact the author.