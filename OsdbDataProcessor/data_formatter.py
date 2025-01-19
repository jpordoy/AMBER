import numpy as np

class DataFormatter:
    def __init__(self, config):
        self.config = config

    def format_data(self, train_df, test_df):
        """
        Format data into the structure expected by the model. This includes reshaping the segments
        and labels and splitting the data into training and testing sets.

        :param train_df: DataFrame for training data.
        :param test_df: DataFrame for testing data.
        :return: Reshaped X_train, X_test, y_train, y_test.
        """
        # Extract segments and labels for training and testing sets
        X_train = np.asarray(train_df['segments'].tolist(), dtype=np.float32)
        y_train = np.asarray(train_df['label'].tolist(), dtype=np.float32)
        X_test = np.asarray(test_df['segments'].tolist(), dtype=np.float32)
        y_test = np.asarray(test_df['label'].tolist(), dtype=np.float32)

        # Reshape the segments to include feature dimension
        X_train_reshaped = self._reshape_segments(X_train)
        X_test_reshaped = self._reshape_segments(X_test)

        return X_train_reshaped, X_test_reshaped, y_train, y_test

    def _reshape_segments(self, segments):
        reshaped_segments = []
        num_samples, num_time_steps, num_features = segments.shape
        for i in range(num_features):
            reshaped_segments.append(segments[:, :, i].reshape(-1, num_time_steps, 1))
        return reshaped_segments
