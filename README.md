# The SKLearn Machine Learning Toolbox

The SKLearn Machine Learning Toolbox is a comprehensive toolchain for data preparation, analysis, model training, and prediction for tabular data. It provides a structured workflow to handle various machine learning tasks, demonstrated with a stock market prediction example.

The toolbox is designed as a template that can be adapted to different datasets and problems. The scripts are organized into a modular structure to guide the user through the machine learning pipeline.

## Project Structure

The project is organized into the following directories:

-   **`src/`**: Contains the Python source code for the machine learning pipeline, divided into the following modules:
    -   `data_generation/`: Scripts for generating ground truth labels and features.
    -   `data_processing/`: Scripts for data cleaning, analysis, feature selection, and data splitting.
    -   `modeling/`: Scripts for hyperparameter tuning, model training, and evaluation.
    -   `prediction/`: Scripts for making predictions and post-processing the results.
    -   `utils/`: Utility functions and helper scripts used across the pipeline.
-   **`tests/`**: Contains unit tests for the scripts in the `src` directory, mirroring the `src` directory structure.
-   **`config/`**: Project-specific configuration files (`.ini`).
-   **`data_raw/`**: Raw input data.
-   **`data_prepared/`**: Processed and cleaned data ready for training.
-   **`models/`**: Saved, trained models.
-   **`results/`**: Output from the pipeline, such as evaluations, predictions, and plots.
-   **`scripts/`**: Batch or shell scripts to execute the pipeline steps.

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

The pipeline is divided into sequential steps, each corresponding to a Python script within the `src` directory.

### Data and Feature Generation (`src/data_generation`)

This step focuses on generating ground truth labels and features from the raw data.

-   **`step20_generate_groundtruth_stockmarket.py`**: Automatically generates labels from the data.
-   **`step20_generate_groundtruth_stockmarket_from_annotation.py`**: Loads manually created labels from a CSV file.
-   **`step21_generate_features.py`**: Generates a full set of features from the raw data.
-   **`step21_generate_features_reduced_lt.py`**: Generates a reduced, specific set of features.
-   **`step22_adapt_dimensions.py`**: Aligns the dimensions of feature and label dataframes.

### Data Preparation, Analysis, and Feature Selection (`src/data_processing`)

This step cleans the data, performs exploratory data analysis, selects the most relevant features, and splits the data for training.

-   **`step30_clean_raw_data.py`**: Cleans the dataset by handling missing values and standardizing column names.
-   **`step31_adapt_features.py`**: Prepares features for machine learning models.
-   **`step32_search_hyperparameters.py`**: A utility script to find optimal hyperparameters for visualization tools.
-   **`step33_analyze_data.py`**: Performs extensive exploratory data analysis (EDA).
-   **`step34_analyze_temporal_data.py`**: Provides tools for analyzing time-series data.
-   **`step35_perform_feature_selection.py`**: Uses various techniques to select the most significant features.
-   **`step36_split_training_validation.py`**: Splits the dataset into training and validation sets.

### Hyperparameter Tuning, Model Training, and Evaluation (`src/modeling`)

This step is for finding the optimal hyperparameters, training the model, and evaluating its performance.

-   **`step42_analyze_training_time.py`**: Analyzes the training time and performance scaling.
-   **`step43_wide_hyperparameter_search_all.py`**: Performs a broad search for the best pipeline components.
-   **`step44_narrow_hyperparameter_search_all.py`**: Conducts a more focused search to fine-tune the model.
-   **`step45_define_precision_recall.py`**: Optimizes the decision threshold for classification models.
-   **`step50_train_model_from_pipe.py`**: Trains the final machine learning model.
-   **`step60_evaluate_model.py`**: Evaluates the model on the validation set.
-   **`step61_evaluate_model_temporal_data.py`**: Provides specific evaluation for time-series data.

### Prediction and Post-processing (`src/prediction`)

This final step uses the trained model to make predictions on new, unseen data.

-   **`step70_predict_temporal_data.py`**: Loads a trained model and uses it to predict outcomes.
-   **`step71_value_postprocessing.py`**: Applies post-processing logic to the raw model predictions.
-   **`step72_backtesting.py`**: Performs a historical simulation to evaluate a trading strategy.

## Configuration

The pipeline is controlled by `.ini` configuration files located in the `config` directory. These files allow you to specify paths, model parameters, and other settings for each step of the process.

## How to Run

The `scripts` directory contains template batch (`.bat`) and shell (`.sh`) scripts for running the pipeline. You can copy and adapt these scripts to your project to execute the steps in the correct order.

## Future Work

-   Integrate more classification and regression models (e.g., Random Forest, KNN, MLP).
-   Expand metrics and visualizations for regression problems.
-   Implement the user-friendliness enhancements outlined in `doc/refactoring2026.md`.

## Issues

For any issues or suggestions, please open an issue on the GitHub repository or contact the author.
