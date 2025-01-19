import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from config import Config as config
from OsdbDataProcessor.osdb_data_label_generator import OsdbDataLabelGenerator
from OsdbDataProcessor.osdb_data_reshaper import OsdbDataReshaper
from OsdbDataProcessor.osdb_interpolator import OsdbInterpolator
from data_loader import DataLoader
from data_formatter import DataFormatter
from model_rf import Amber_RF
from kfold_cv import KFoldCrossValidation
from model_evaluator import ModelEvaluator


class SeizurePredictior:
    def __init__(self, file_path, priority_test_event_ids):
        self.file_path = file_path
        self.priority_test_event_ids = priority_test_event_ids

        # Process data
        self.processor = OsdbDataLabelGenerator(self.file_path)
        self.df_result = self.processor.process_data()

        # Reshape data
        self.data_reshaper = OsdbDataReshaper(self.df_result)
        self.reshaped_df = self.data_reshaper.reshape_data()

        # Interpolate HR data
        self.interpolator = OsdbInterpolator(self.reshaped_df, column_to_interpolate="hr")
        self.interpolator.interpolate_column(new_column_name="interpolated_hr", interval=config.N_TIME_STEPS, time_step=config.time_step_length)
        self.df_sensor_data = self.interpolator.get_dataframe()

        # Load data
        self.data_loader = DataLoader(dataframe=self.df_sensor_data, time_steps=config.N_TIME_STEPS, step=config.step, target_column='label')
        self.df_labels = self.data_loader.load_data()

        # Format data
        self.data_formatter = DataFormatter(config)
        self.X_train_reshaped, self.X_test_reshaped, self.y_train, self.y_test = self.data_formatter.format_data(self.df_labels, self.priority_test_event_ids)

        # Initialize model
        self.model_rf = Amber_RF(row_hidden=config.row_hidden, col_hidden=config.row_hidden, num_classes=2)
        
        # Reshape y_test correctly
        self.y_test_reshaped = np.asarray(self.y_test, dtype=np.float32)

    def train_and_evaluate(self):
        # Create an instance of KFoldCrossValidation
        kfold_cv = KFoldCrossValidation(self.model_rf, [self.X_train_reshaped['Feature_1'], self.X_train_reshaped['Feature_2'], self.X_train_reshaped['Feature_3']], self.y_train)
        
        # Run cross-validation
        kfold_cv.run()

        # Evaluate the model
        evaluator = ModelEvaluator(self.model_rf, [self.X_test_reshaped['Feature_1'], self.X_test_reshaped['Feature_2'], self.X_test_reshaped['Feature_3']], self.y_test_reshaped)
        evaluator.evaluate()

        # Get evaluation results
        evaluation_results = evaluator.get_evaluation_results()
        print("Accuracy:", evaluation_results["accuracy"])
        print("F1 Score:", evaluation_results["f1"])
        print("Classification Report:\n", evaluation_results["classification_report"])

        # Predict test data
        y_pred = self.model_rf.model.predict([self.X_test_reshaped['Feature_1'], self.X_test_reshaped['Feature_2'], self.X_test_reshaped['Feature_3']])
        return y_pred

    def plot_metrics(self, y_pred):
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)

        # Filter the test DataFrame by the priority event IDs
        df_test = self.df_labels[self.df_labels['eventId'].isin(self.priority_test_event_ids)]

        # Create a DataFrame for the test results
        test_results_df = pd.DataFrame({
            'eventId': df_test['eventId'].values,
            'userId': df_test['userId'].values,
            'y_true': y_true_classes,
            'y_pred': y_pred_classes
        })

        # Calculate metrics for each group
        def calculate_metrics_for_group(group):
            cm = confusion_matrix(group['y_true'], group['y_pred'], labels=[0, 1])
            accuracy = accuracy_score(group['y_true'], group['y_pred'])
            TP = np.diag(cm)
            FP = cm.sum(axis=0) - TP
            FN = cm.sum(axis=1) - TP
            TN = cm.sum() - (TP + FP + FN)

            TPR = TP / (TP + FN)  # True Positive Rate (Sensitivity)
            FPR = FP / (FP + TN)  # False Positive Rate
            FAR = FP + FN / (TP + FN + TN + FP)  # False Alarm Rate
            FNR = FN / (TP + FN)  # False Negative Rate
            TNR = TN / (TN + FP)  # True Negative Rate (Specificity)
            PPV = TP / (TP + FP)  # Positive Predictive Value (Precision)
            NPV = TN / (TN + FN)  # Negative Predictive Value
            FDR = FP / (TP + FP)  # False Discovery Rate
            return pd.Series({
                'accuracy': accuracy,
                'TPR': TPR.mean(),
                'FPR': FPR.mean(),
                'FAR': FAR.mean(),
                'FNR': FNR.mean(),
                'TP': TP,
                'FP': FP,
                'TN': TN,
                'FN': FN,
                'TNR': TNR.mean(),
                'PPV': PPV.mean(),
                'NPV': NPV.mean(),
                'FDR': FDR.mean()
            })

        # Group by userID and eventID to calculate metrics
        user_metrics = test_results_df.groupby('userId').apply(calculate_metrics_for_group).reset_index()
        event_metrics = test_results_df.groupby('eventId').apply(calculate_metrics_for_group).reset_index()

        # Plot grouped bar chart for userID metrics (Accuracy and TPR)
        self._plot_user_metrics(user_metrics)

        # Plot grouped bar chart for userID metrics (FAR, FPR, FNR)
        self._plot_user_metrics_fpr_fnr(user_metrics)

        # Plot grouped bar chart for eventID metrics (Accuracy and TPR)
        self._plot_event_metrics(event_metrics)

        # Plot grouped bar chart for eventID metrics (FAR, FPR, FNR)
        self._plot_event_metrics_fpr_fnr(event_metrics)

    def _plot_user_metrics(self, user_metrics):
        fig, ax = plt.subplots(figsize=(10, 3))  # Increased figure size for better visibility
        width = 0.15  # Reduced width of the bars

        x = np.arange(len(user_metrics['userId']))  # x locations for the groups

        # Plotting the bars
        ax.bar(x - width * 1.5, user_metrics['accuracy'], width, label='Accuracy', align='center')
        ax.bar(x - width / 2, user_metrics['TPR'], width, label='TPR (Sensitivity)', align='center')
        ax.bar(x + width / 2, user_metrics['TNR'], width, label='TNR', align='center')
        ax.bar(x + width * 1.5, user_metrics['PPV'], width, label='PPV', align='center')

        # Formatting the plot
        ax.set_xlabel('UserID')
        ax.set_ylabel('Score')
        ax.set_title('Accuracy, Sensitivity (TPR), TNR, and PPV for Each UserID')
        ax.set_xticks(x)
        ax.set_xticklabels(user_metrics['userId'], rotation=45, ha='right')  # Adjust rotation for better readability
        ax.legend()
        plt.tight_layout()
        plt.grid(axis='y')  # Optionally add gridlines for easier reading
        plt.show()

    def _plot_user_metrics_fpr_fnr(self, user_metrics):
        fig, ax = plt.subplots(figsize=(9, 3))

        width = 0.15  # Reduced width of the bars
        x = np.arange(len(user_metrics['userId']))

        ax.bar(x - width, user_metrics['FAR'], width, label='FAR (False Alarm Rate)')
        ax.bar(x, user_metrics['FPR'], width, label='FPR (False Positive Rate)')
        ax.bar(x + width, user_metrics['FNR'], width, label='FNR (False Negative Rate)')

        # Formatting the plot
        ax.set_xlabel('UserId')
        ax.set_ylabel('Score')
        ax.set_title('FAR, FPR, and FNR for each UserID')
        ax.set_xticks(x)
        ax.set_xticklabels(user_metrics['userId'], rotation=90)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _plot_event_metrics(self, event_metrics):
        fig, ax = plt.subplots(figsize=(9, 3))
        x = np.arange(len(event_metrics['eventId']))  # Update x for eventID
        ax.bar(x - 0.15, event_metrics['accuracy'], 0.3, label='Accuracy')
        ax.bar(x, event_metrics['TPR'], 0.3, label='TPR (Sensitivity)')

        # Formatting the plot
        ax.set_xlabel('EventID')
        ax.set_ylabel('Score')
        ax.set_title('Accuracy and Sensitivity (TPR) for each EventID')
        ax.set_xticks(x)
        ax.set_xticklabels(event_metrics['eventId'], rotation=90)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _plot_event_metrics_fpr_fnr(self, event_metrics):
        fig, ax = plt.subplots(figsize=(9, 3))
        x = np.arange(len(event_metrics['eventId']))  # Update x for eventID
        ax.bar(x - 0.15, event_metrics['FAR'], 0.3, label='FAR (False Alarm Rate)')
        ax.bar(x, event_metrics['FPR'], 0.3, label='FPR (False Positive Rate)')
        ax.bar(x + 0.15, event_metrics['FNR'], 0.3, label='FNR (False Negative Rate)')

        # Formatting the plot
        ax.set_xlabel('EventID')
        ax.set_ylabel('Score')
        ax.set_title('FAR, FPR, and FNR for each EventID')
        ax.set_xticks(x)
        ax.set_xticklabels(event_metrics['eventId'], rotation=90)
        ax.legend()
        plt.tight_layout()
        plt.show()


