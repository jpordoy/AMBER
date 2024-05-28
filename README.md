# AMBER MODEL
Author: Dr Jamie Pordoy

---

## Introduction
AMBER (Attention-guided Multi-Branching-pipeline with Enhanced Residual fusion) is a deep learning architecture designed for multimodal seizure detection tasks. It addresses the challenges of processing heterogeneous data sources by constructing dedicated branches for each feature modality. 

We extend our gratitude to the open-source community, whose contributions have significantly aided the development and dissemination of our work. While the coding style in this repository is still in its early stages, we welcome and encourage contributions to help refactor and improve its efficiency and readability. We appreciate the collective effort that makes advancements in multimodal detection and classification possible.)

---
![Federated Transfer Learning Framework architecture](Images/AMBER.png)
---


## Network Architecture


Our proposed architecture comprises the following four layers: the manifold reduction layer, the common embedded space, the tangent projection layer, and the federated layer. The function of each layer is detailed below:

1. **Manifold Reduction Layer**: Spatial covariance matrices are consistently presumed to inhabit high-dimensional Symmetric Positive Definite (SPD) manifolds. This layer acts as a linear map from the high-dimensional SPD manifold to the low-dimensional counterpart, with undefined weights reserved for learning.

2. **Common Embedded Space**: The common space is the low-dimensional SPD manifold, whose elements are diminished from each high-dimensional SPD manifold. It is specifically designed for the transfer learning setting.

3. **Tangent Projection Layer**: The function of this layer is to project the matrices on SPD manifolds to its tangent space, which is a local linear approximation of the curved space.

4. **Federated Layer**: Deep neural networks are implemented in this layer. For the transfer learning setting, the parameters of neural networks are updated by federated aggregation.

---
![Federated Transfer Learning Framework architecture](https://github.com/jpordoy/AMBER/blob/master/Images/model_plot.png)


### How To Run The Code
Please put your training data and labels into a directory "raw_data/" in this project.
The package `mne` is adopted for EEG data pro-processing. To generate the required data as SPDNet input, please refer to the following example code: 

```python        
import pandas as pd
import numpy as np
from data_loader import DataLoader
from data_formatter import DataFormatter
from model import Amber
from kfold_cv import KFoldCrossValidation
from evaluator import evaluate_model_performance
from config import config

# Define your DataFrame and parameter
mypath = 'Data/Train.csv'
df = pd.read_csv(mypath)
target_column = 'label'  # Name of the target column

# Step 1: Load Data
data_loader = DataLoader(dataframe=df, time_steps=config.N_TIME_STEPS, step=config.step, target_column=target_column)
segments, labels = data_loader.load_data()

# Step 2: Format Data
data_formatter = DataFormatter(config=config)
X_train_reshaped, X_test_reshaped, y_train, y_test = data_formatter.format_data(segments, labels)

# Reshape y_test correctly
y_test_reshaped = np.asarray(y_test, dtype=np.float32)

# Initialize model
ts_model = Amber(row_hidden=config.row_hidden, col_hidden=config.row_hidden, num_classes=config.N_CLASSES)

# Create an instance of KFoldCrossValidation
kfold_cv = KFoldCrossValidation(ts_model, [X_train_reshaped['Feature_1'], X_train_reshaped['Feature_2']], y_train)

# Run the cross-validation
kfold_cv.run()

# Evaluate the model performance
evaluation_results = evaluate_model_performance(ts_model, [X_test_reshaped['Feature_1'], X_test_reshaped['Feature_2']], y_test_reshaped)

# Access individual metrics
print("Accuracy:", evaluation_results["accuracy"])
print("F1 Score:", evaluation_results["f1"])
print("Cohen's Kappa:", evaluation_results["cohen_kappa"])

```

### Model training

For subject-adaptive analysis, run `SPDNet_Federated_Transfer_Learning.py `

For subject-specific analysis, run `SPDNet_Local_Learning.py`


