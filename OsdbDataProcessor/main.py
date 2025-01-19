import sys
import os
import warnings
import numpy as np
import logging
import tensorflow as tf
from config import Config
from osdb_interpolator import OsdbInterpolator
from osdb_data_label_generator import OsdbDataLabelGenerator
from osdb_data_reshaper import OsdbDataReshaper
from data_loader import DataLoader
from data_formatter import DataFormatter
from event_based_splitter import EventBasedSplitter
from model import Amber
from k_fold_cross_validation import KFoldCrossValidation
from model_evaluator import evaluate_model_performance
from model_tester import ModelTester
import pandas as pd
# Ensure the current directory is added to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Suppress specific FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning, message=".*mode.*keepdims.*")


# Configure logging
log_filename = 'app_log.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO logs
tf.get_logger().setLevel('ERROR')

# Custom TensorFlow log handler
class TensorFlowLogHandler(logging.Handler):
    def emit(self, record):
        try:
            log_message = self.format(record)
            logger.error(log_message)  # Redirect TensorFlow logs as errors
        except Exception:
            self.handleError(record)

tensorflow_handler = TensorFlowLogHandler()
tensorflow_handler.setLevel(logging.ERROR)
tf.get_logger().addHandler(tensorflow_handler)

# Main function
def main(file_path):
    try:
        
        
        # Step 1: Load labeled data
        logger.info("Step 1: Establishing connection to OSDB.")
        if not os.path.exists(file_path):
            logger.error(f"Critical error: Specified file not found at {file_path}. Aborting process.")
            return

        try:
            processor = OsdbDataLabelGenerator(file_path)
            df_result = processor.process_data()
        except Exception as e:
            logger.error(f"Data processing error: Failed to load and process OSDB data. Reason: {e}", exc_info=True)
            return

        if df_result.empty:
            logger.error("Data error: OSDB connection established, but returned dataset is empty.")
            return
        logger.info("Connection to OSDB established and data loaded successfully.")



        # Step 2: Reshape the flattened DataFrame
        logger.info("Step 2: Reshaping raw dataset for further processing.")
        try:
            reshaper = OsdbDataReshaper(df_result)
            reshaped_df = reshaper.reshape_data()
        except Exception as e:
            logger.error(f"Reshaping error: Failed to reshape the dataset. Reason: {e}", exc_info=True)
            return

        if reshaped_df.empty:
            logger.error("Transformation error: Data reshaping failed, resulting DataFrame is empty.")
            return
        logger.info("Dataset reshaped successfully.")



        # Step 3: Interpolate missing data
        logger.info("Step 3: Performing interpolation to fill missing values in dataset.")
        try:
            interpolator = OsdbInterpolator(reshaped_df, column_to_interpolate="hr")
            interpolator.interpolate_column(
                new_column_name="interpolated_hr",
                interval=Config.N_TIME_STEPS,
                time_step=Config.length_time_step,
            )
            dataset_df = interpolator.get_dataframe()
        except Exception as e:
            logger.error(f"Interpolation error: Failed to interpolate dataset. Reason: {e}", exc_info=True)
            return

        if dataset_df.empty:
            logger.error("Processing error: Interpolation failed, resulting DataFrame is empty.")
            return
        logger.info("Interpolation completed successfully.")



        # Step 4: Structure data into time steps
        logger.info("Step 4: Structuring dataset into time steps and set features")
        try:
            # Define the feature columns
            feature_columns = ['rawData', 'interpolated_hr', 'FFT']  # You can modify these features to add or remove

            # Initialize the DataLoader with the new dynamic feature list
            data_loader = DataLoader(
                dataframe=dataset_df,
                time_steps=Config.N_TIME_STEPS,
                step=Config.step,
                target_column="label",
                feature_columns=feature_columns
            )
            df_label = data_loader.load_data()
        except Exception as e:
            logger.error(f"Time step structuring error: Failed to structure dataset. Reason: {e}", exc_info=True)
            return
        logger.info("Dataset successfully structured into time steps.")



        # Step 5: Split dataset by event ID
        logger.info("Step 5: Splitting dataset into training and testing subsets by event ID.")
        try:
            splitter = EventBasedSplitter(dataframe=df_label, test_size=0.25)
            train_df, test_df = splitter.split_by_event()
        except Exception as e:
            logger.error(f"Splitting error: Failed to split dataset. Reason: {e}", exc_info=True)
            return
        logger.info("Dataset split completed successfully into training and testing subsets.")



        # Step 6: Format data for model input
        logger.info("Step 6: Formatting dataset for compatibility with the AMBER model.")
        try:
            data_formatter = DataFormatter(Config)
            X_train_reshaped, X_test_reshaped, y_train, y_test = data_formatter.format_data(train_df, test_df)
        except Exception as e:
            logger.error(f"Formatting error: Failed to format dataset. Reason: {e}", exc_info=True)
            return
        logger.info("Dataset successfully formatted for AMBER model input.")



        # Step 7: Initialize model and perform training with cross-validation
        logger.info("Step 7: Initializing AMBER model and starting training with K-Fold Cross Validation.")
        try:
            ts_model = Amber(row_hidden=Config.row_hidden, col_hidden=Config.row_hidden, num_classes=Config.N_CLASSES)
            logger.info("AMBER model initialized.")
        except Exception as e:
            logger.error(f"Model initialization error: Failed to initialize AMBER model. Reason: {e}", exc_info=True)
            return

        # Extract features for training and testing dynamically
        try:
            X_train_features = [X_train_reshaped[i] for i in range(len(feature_columns))]
            X_test_features = [X_test_reshaped[i] for i in range(len(feature_columns))]
            y_test_reshaped = np.asarray(y_test, dtype=np.float32)
        except Exception as e:
            logger.error(f"Feature extraction error: Failed to extract features. Reason: {e}", exc_info=True)
            return

        # Start K-Fold Cross Validation with dynamic features
        try:
            kfold_cv = KFoldCrossValidation(
                ts_model,
                X_train_features,  # Pass all features dynamically
                y_train,
            )
            logger.info("Training started.")
            kfold_cv.run()
            logger.info("K-Fold Cross Validation completed. All folds processed successfully.")
        except Exception as e:
            logger.error(f"Training error: K-Fold Cross Validation failed. Reason: {e}", exc_info=True)
            return


        # Step 8: Save Model
        logger.info("Step 8: Save model.")
        try:
            model_save_path = os.path.join("Models", "AMBER_Model.keras")
            ts_model.save_model(path=model_save_path)
        except Exception as e:
            logger.error(f"Saving model error: Failed to save AMBER model. Reason: {e}", exc_info=True)



        # Step 9: Load Model
        logger.info("Step 9: Load model.")
        try:
            model_save_path = os.path.join("Models", "AMBER_Model.keras")
            ts_model = Amber.load_model(path=model_save_path)  # Calling the static method
        except Exception as e:
            logger.error(f"Saving model error: Failed to load AMBER model. Reason: {e}", exc_info=True)
        
        
        
        # Step 10: Evaluate model performance
        logger.info("Step 10: Evaluating model performance on the test dataset.")
        try:
            evaluation_results = evaluate_model_performance(
                ts_model,
                X_test_features,
                y_test_reshaped,
            )
        except Exception as e:
            logger.error(f"Evaluation error: Model evaluation failed. Reason: {e}", exc_info=True)
            return

        logger.info("Model evaluation completed. Performance metrics:")
        logger.info(f"- Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"- F1 Score: {evaluation_results['f1']:.4f}")
        

    
        
        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.error(f"Unhandled exception occurred: {e}", exc_info=True)
        
        


# Path to JSON file
json_file_path = os.path.join(os.path.dirname(__file__), "..", "Data", "osdb_3min_allSeizures.json")
main(json_file_path)
