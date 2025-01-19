<div>
<img src="https://github.com/jpordoy/AMBER/tree/osdb_branch/Images/3.png" alt="logo" width="200" height="auto" />

<p>
AMBER (Attention-guided Multi-Branching-pipeline with Enhanced Residual fusion) is an experimental deep learning architecture for biomedical engineering. Designed for one-dimensional, multimodal detection tasks, the architecture addresses the challenges of processing heterogeneous data sources by constructing independent pipelines for each feature modality
</p>
  
</div>

<br />

---
### 📂 Repository Structure

```plaintext

AMBER/
│
├── app_log.log               # Application logs
├── config.py                 # Configuration file with parameters
├── data_loader.py            # Data loading utilities
├── data_formatter.py         # Data formatting utilities
├── event_metrics_evaluator.py # Event metrics evaluation
├── kfold_cv.py               # K-fold cross-validation logic
├── model_rf.py               # Residual fusion model definition
├── model_evaluator.py        # Model evaluation utilities
├── OsdbDataProcessor/        # Data Loading preprocessing and annotation for the OSDB
│   ├── osdb_data_label_generator.py
│   ├── osdb_data_reshaper.py
│   └── osdb_interpolator.py
├── main.py                   # Main entry point for the pipeline
└── requirements.txt          # List of required Python packages

```
---

### Setup Instructions

#### 1. Clone the Repository 📂
Clone the repository using Git:

```bash
git clone https://github.com/jpordoy/AMBER.git
cd AMBER
```
---

#### 2. Create and Activate a Virtual Environment
Ensure you're using Python 3.8 for the project. You can create a virtual environment using the following commands:

#### For Linux/MacOS:
```bash
python3.8 -m venv venv
source venv/bin/activate
```
#### For Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies
Install the required dependencies using pip and the requirements.txt file provided:
```bash
Copy code
pip install -r requirements.txt
```
---

### ▶️ How To Run The Code
Please put your training data as a csv file in the "Data/" of this project.

```python        
if __name__ == "__main__":
    
    # Example usage:
    # Instructions for processing the data, reshaping it, and interpolating the 'hr' column
    # can be found in the OsdbDataProcessor component for the OSDB.

    # Ensure you follow the steps in the OSDBDataProcessor for:
    # 1. Data processing using the OsdbLabelGenerator
    # 2. Data reshaping with the DataReshaper
    # 3. Interpolation of the 'hr' column with the Interpolator

    # Initialize DataLoader
    data_loader = DataLoader(dataframe=interpolated_df, time_steps=config.N_TIME_STEPS, step=config.step, target_column='label')
    # Load data (this will return a DataFrame with segments, labels, eventID, and userID)
    df_labels = data_loader.load_data()

    # Specify the test event IDs, otherwise split by Random
    priority_test_event_ids = [5595, 5596, ... ,5610]

    # Initialize DataFormatter
    data_formatter = DataFormatter(config)

    # Split the data by eventID into train and test sets
    X_train_reshaped, X_test_reshaped, y_train, y_test = data_formatter.format_data(df_labels, priority_test_event_ids)
    # Reshape y_test correctly
    y_test_reshaped = np.asarray(y_test, dtype=np.float32)
    
    # Initialize model with residual fusion layer
    ts_model = Amber_RF(row_hidden=config.row_hidden, col_hidden=config.row_hidden, num_classes=2)
    # Build the model    
    
    # Create an instance of KFoldCrossValidation
    kfold_cv = KFoldCrossValidation(ts_model, [X_train_reshaped['Feature_1'], X_train_reshaped['Feature_2'], X_train_reshaped['Feature_3']], y_train)

    # Run the cross-validation
    kfold_cv.run()
    
    # Evaluate the model performance
    evaluation_results = evaluate_model_performance(ts_model, [X_test_reshaped['Feature_1'], X_test_reshaped['Feature_2'], X_test_reshaped['Feature_3']], y_test_reshaped)
    print("\nOverall Classification Results\n")
    # Access individual metrics
    print("Accuracy:", evaluation_results["accuracy"])
    print("F1 Score:", evaluation_results["f1"])
```
---


