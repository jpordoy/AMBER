import warnings
import numpy as np
import logging
import tensorflow as tf
from config import Config as config
from OsdbDataProcessor.osdb_data_label_generator import OsdbDataLabelGenerator
from OsdbDataProcessor.osdb_data_reshaper import OsdbDataReshaper
from OsdbDataProcessor.osdb_interpolator import OsdbInterpolator
from data_loader import DataLoader
from data_formatter import DataFormatter
from model_rf import Amber_RF
from kfold_cv import KFoldCrossValidation
from model_evaluator import ModelEvaluator
from event_metrics_evaluator import EventMetricsEvaluator

# Suppress specific FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning, message=".*mode.*keepdims.*")

# Configure logging
log_filename = 'app_log.log'  # Specify your log file name
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Logs to console
                        logging.FileHandler(log_filename)  # Logs to file
                    ])
logger = logging.getLogger(__name__)

# Suppress TensorFlow info and warnings
tf.get_logger().setLevel('ERROR')

# Create a custom logging handler to redirect TensorFlow logs
class TensorFlowLogHandler(logging.Handler):
    def emit(self, record):
        try:
            log_message = self.format(record)
            logger.error(log_message)  # Log TensorFlow messages as errors
        except Exception:
            self.handleError(record)

# Add the custom TensorFlow log handler to capture logs
tensorflow_handler = TensorFlowLogHandler()
tensorflow_handler.setLevel(logging.ERROR)
tf.get_logger().addHandler(tensorflow_handler)

def main():
    try:
        # Set the file path for the dataset
        file_path = 'Data/osdb_3min_allSeizures.json'  # Replace with your JSON file path
        logger.info(f"Loading data from {file_path}")
        
        # Initialize OsdbDataLabelGenerator and process data
        processor = OsdbDataLabelGenerator(file_path)
        df_result = processor.process_data()
        logger.info("Data processing complete.")
        
        # Reshape the data
        data_reshaper = OsdbDataReshaper(df_result)
        reshaped_df = data_reshaper.reshape_data()
        logger.info("Data reshaping complete.")
        
        # Initialize Interpolator and interpolate the 'hr' column
        interpolator = OsdbInterpolator(reshaped_df, column_to_interpolate="hr")
        interpolator.interpolate_column(new_column_name="interpolated_hr", interval=config.N_TIME_STEPS, time_step=config.time_step_length)
        df_sensor_data = interpolator.get_dataframe()
        logger.info("Data interpolation complete.")
        
        # Load the data using DataLoader
        data_loader = DataLoader(dataframe=df_sensor_data, time_steps=config.N_TIME_STEPS, step=config.step, target_column='label')
        df_labels = data_loader.load_data()
        logger.info("Data loading complete.")
        
        # Specify the priority test event IDs
        priority_test_event_ids = [
            5595, 5596, 28725, 28734, 40913, 14101, 15208, 26071, 26077, 26988,
            21603, 21695, 21797, 21855, 15039,
            12618, 12624, 12763, 5635, 5637, 6668, 8726,
            7219, 7222, 6732, 5721, 7258, 7262, 6761, 5745,
            5254, 7823, 11591, 40784, 5610
        ]
        
        # Initialize DataFormatter and split the data by eventID
        data_formatter = DataFormatter(config)
        X_train_reshaped, X_test_reshaped, y_train, y_test = data_formatter.format_data(df_labels, priority_test_event_ids)
        
        # Reshape y_test correctly
        y_test_reshaped = np.asarray(y_test, dtype=np.float32)
        logger.info("Data formatting complete.")
        
        # Print reshaped y_test
        logger.info(f"y_test reshaped complete.")
        
        # Initialize model with residual fusion layer
        model_rf = Amber_RF(row_hidden=config.row_hidden, col_hidden=config.row_hidden, num_classes=2)
        logger.info("Model initialized.")
        
        # Create an instance of KFoldCrossValidation and run cross-validation
        kfold_cv = KFoldCrossValidation(model_rf, [X_train_reshaped['Feature_1'], X_train_reshaped['Feature_2'], X_train_reshaped['Feature_3']], y_train)
        kfold_cv.run()
        logger.info("Cross-validation complete.")
        
        # Initialize evaluator and evaluate the model
        evaluator = ModelEvaluator(model_rf, [X_test_reshaped['Feature_1'], X_test_reshaped['Feature_2'], X_test_reshaped['Feature_3']], y_test_reshaped)
        evaluator.evaluate()
        logger.info("Model evaluation complete.")
        
        # Get the evaluation results
        evaluation_results = evaluator.get_evaluation_results()
        logger.info(f"Accuracy: {evaluation_results['accuracy']}")
        logger.info(f"F1 Score: {evaluation_results['f1']}")
        logger.info(f"Classification Report:\n{evaluation_results['classification_report']}")
        
        logger.info("Tasks Completed")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
