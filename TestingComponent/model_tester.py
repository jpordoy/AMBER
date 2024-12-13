import pandas as pd
import numpy as np
import tensorflow as tf
import random

class ModelTester:
    def __init__(self, model, time_steps):
        self.model = model
        self.time_steps = time_steps

    @staticmethod
    def filter_by_eventID(df, event_id):
        filtered_df = df[df['eventID'] == event_id]
        return filtered_df

    def preprocess_single_timestep(self, data):
        rawData = data[['rawData']].values.reshape(1, self.time_steps, 1)
        hr = data[['ppg']].values.reshape(1, self.time_steps, 1)
        actual_label = data['label'].iloc[0]  # Extract the actual label
        return rawData, hr, actual_label

    def predict(self, dataframe):
        if len(dataframe) < self.time_steps:
            raise ValueError("DataFrame does not have enough rows for the required time steps.")

        # Randomly select a starting index for the time step
        start_index = random.randint(0, len(dataframe) - self.time_steps)
        single_time_step_data = dataframe.iloc[start_index:start_index + self.time_steps]

        reshaped_segments, reshaped_hr, actual_label = self.preprocess_single_timestep(single_time_step_data)
        prediction = self.model.predict([reshaped_segments, reshaped_hr])
        predicted_class = int(np.argmax(prediction))

        return predicted_class, prediction, actual_label

    def batch_predict(self, dataframe, step=1):
        results = []
        # Total number of predictions that can be made based on the dataframe length and time steps
        num_batches = len(dataframe) // self.time_steps

        for i in range(num_batches):
            start_index = i * self.time_steps
            all_event_data = dataframe.iloc[start_index:start_index + self.time_steps]
            reshaped_segments, reshaped_hr, actual_label = self.preprocess_single_timestep(all_event_data)
            prediction = self.model.predict([reshaped_segments, reshaped_hr])

            predicted_class = int(np.argmax(prediction))
            results.append({
                'start_index': start_index,
                'predicted_class': predicted_class,
                'probability_distribution': prediction.flatten().tolist(),
                'actual_label': actual_label
            })

        return pd.DataFrame(results)

    def visualize_shapes(self, data, rawData, hr, prediction):
        # Visualize the original data and reshaped input
        print("Original data shape:", data.shape)
        print("Acceleration (RawData) shape:", rawData.shape)
        print("HR shape:", hr.shape)
        print("Prediction shape:", prediction.shape)


def main():
    # Path to the dataset
    path = "testing_dataset.csv"
    df = pd.read_csv(path)
    print("Dataset loaded successfully!")

    # Load the trained model
    model_path = 'my_testing_model.keras'
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

    # Initialise the ModelTester with the trained model
    model_tester = ModelTester(model=model, time_steps=125)

    # Test filtered event classification
    filtered_data = model_tester.filter_by_eventID(df, 5635)
    results_df = model_tester.batch_predict(filtered_data)
    print("*** Filtered Event Classification Test ***")
    print(results_df.head())

    # Test single time-step classification
    print("*** Single Timestep Classification Test ***")
    predicted_class, probability_distribution, actual_label = model_tester.predict(df)
    print("Predicted class:", predicted_class)
    print("Probability distribution:", probability_distribution)
    print("Actual label:", actual_label)

    # Visualize the shapes after all predictions
    print("*** Visualizing Input/Output Shapes ***")
    # Visualize a sample (the first segment in this case)
    reshaped_segments, reshaped_hr, _ = model_tester.preprocess_single_timestep(df.iloc[0:125])
    prediction = model.predict([reshaped_segments, reshaped_hr])
    model_tester.visualize_shapes(df.iloc[0:125], reshaped_segments, reshaped_hr, prediction)


if __name__ == "__main__":
    main()
