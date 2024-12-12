# Seizure Event Classification

This component tests AMBER models in classifying time-series data related to **Open Seizure Databases** using TensorFlow. The `ModelTester` class is responsible for loading the trained model, performing predictions on both individual time steps and entire events, and visualising the input/output shapes.

---

## :rocket: **Project Overview**

The **`ModelTester`** class contains methods for:
- Preprocessing data
- Making predictions for time steps and events
- Visualising input/output shapes

---

## :memo: **Functions and Their Purpose**

### 1. **`ModelTester` Class**

The `ModelTester` class contains methods for preprocessing the data, making predictions, and visualising the results.

#### **`__init__(self, model, time_steps)`**
Initialises the class with the model and time steps.

- `model`: The trained machine learning model (e.g., Keras model).
- `time_steps`: Number of time steps the model requires for classification (e.g., 125).

---

#### **`filter_by_eventID(df, event_id)`**
Filters the DataFrame by a specific `eventID`.

- `df`: The input DataFrame containing the event data.
- `event_id`: The event ID to filter by.

**Returns**: A DataFrame containing only the rows where the `eventID` matches the specified value.

---

#### **`preprocess_single_timestep(data)`**
Preprocesses a single time step (i.e., a sequence of data) for input into the model. This reshapes the `rawData` and `ppg` columns into the required input format.

- `data`: DataFrame containing one time step.
  
**Returns**:
- `segments`: Reshaped segment data (3D array of shape `(1, time_steps, 1)`).
- `hr`: Reshaped heart rate data (3D array of shape `(1, time_steps, 1)`).
- `actual_label`: The true label of the data point.

---

#### **`predict(dataframe)`**
Makes a prediction for a randomly selected time step from the input DataFrame. It preprocesses the data, performs prediction using the model, and returns the predicted class and probability distribution.

- `dataframe`: The input DataFrame containing the full dataset.
  
**Returns**:
- `predicted_class`: The class predicted by the model.
- `probability_distribution`: The predicted probability distribution for all classes.
- `actual_label`: The true label of the selected data point.

---

#### **`all_event_prediction(dataframe, step=1)`**
Makes predictions for all events in the input DataFrame by processing the data in batches.

- `dataframe`: The DataFrame containing event data.
- `step`: Step size for iterating through the data (default is `1`).

**Returns**: A DataFrame with:
- `start_index`: The starting index of the batch.
- `predicted_class`: The predicted class for the batch.
- `probability_distribution`: The predicted probability distribution for the batch.
- `actual_label`: The true label for the batch.

---

#### **`visualise_shapes(data, segments, hr, prediction)`**
Visualises the shapes of the original data, reshaped segments, heart rate data, and model prediction.

- `data`: The original raw data.
- `segments`: The reshaped segment data.
- `hr`: The reshaped heart rate data.
- `prediction`: The model's prediction output.

---

### 2. **`main()`**
The main function that drives the process:

1. Loads the dataset from a CSV file.
2. Loads the trained model.
3. Initializes the `ModelTester` object.
4. Performs:
   - Event-based classification using `all_event_prediction`.
   - Single time-step classification using `predict`.
   - Visualizes input/output shapes using `visualise_shapes`.

---

## :wrench: **How to Use the Code**

### **1. Loading the Dataset**
Replace the `path` variable with the path to your dataset file. Ensure the dataset is in CSV format and contains columns such as `rawData`, `ppg`, `label`, and `eventID`.

```python
path = "CSV file path"
df = pd.read_csv(path)

