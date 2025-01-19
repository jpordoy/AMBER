import pandas as pd 
import numpy as np 
import os 

class EventMetricsEvaluator:
    def __init__(self, model, events_folder, n_time_steps=125):
        self.model = model
        self.events_folder = events_folder
        self.n_time_steps = n_time_steps
        self.metrics_summary = pd.DataFrame(columns=['CSV_File', 'eventId', 'userId', 'Accuracy', 'Sensitivity', 'False_Positive_Rate', 'False_Negative_Rate', 'False_Alarm_Rate'])
        self.user_event_counts = {}
        self.user_metrics = {}
        self.event_metrics = []

    def calculate_metrics(self, y_true, y_pred):
        TP, TN, FP, FN = 0, 0, 0, 0

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label in [1, 2]:
                if pred_label in [1, 2]:
                    TP += 1
                else:
                    FN += 1
            elif true_label == 0:
                if pred_label in [1, 2]:
                    FP += 1
                else:
                    TN += 1

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        false_positive_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
        false_negative_rate = FN / (TP + FN) if (TP + FN) > 0 else 0
        false_alarm_rate = FP / len(y_true) if len(y_true) > 0 else 0

        return sensitivity, false_positive_rate, false_negative_rate, false_alarm_rate

    def evaluate(self):
        csv_files = [f for f in os.listdir(self.events_folder) if f.endswith('.csv')]

        for csv_file in csv_files:
            mypath = os.path.join(self.events_folder, csv_file)
            df = pd.read_csv(mypath).fillna(-1)

            segments_acceleration = []
            segments_heart_rate = []
            labels = []
            eventIds = []
            userIds = []
            false_alarms = 0

            for i in range(0, len(df), self.n_time_steps):
                group = df.iloc[i:i + self.n_time_steps]
                if len(group) == self.n_time_steps:
                    segment_acceleration = group['rawData'].values
                    segment_heart_rate = group['hr'].values

                    label = group['label'].mode()[0]
                    eventIds.append(group['eventId'].values[0])
                    userIds.append(group['userId'].values[0])

                    segments_acceleration.append(segment_acceleration)
                    segments_heart_rate.append(segment_heart_rate)
                    labels.append(label)

            # Convert segments to numpy arrays
            segments_acceleration = np.array(segments_acceleration, dtype=np.float32).reshape(-1, self.n_time_steps, 1)
            segments_heart_rate = np.array(segments_heart_rate, dtype=np.float32).reshape(-1, self.n_time_steps, 1)

            # Model predictions
            preds = self.model.predict([segments_acceleration, segments_heart_rate])
            preds_classes = np.argmax(preds, axis=-1) if preds.ndim > 1 else preds

            # Ground truth labels
            ground_truth_labels = np.array(labels)

            # False alarms calculation
            for true_label, pred_label in zip(ground_truth_labels, preds_classes):
                if true_label in [0, 2] and pred_label == 1:
                    false_alarms += 1

            total = len(preds_classes)
            overall_false_alarm_rate = false_alarms / total if total > 0 else 0

            # Calculate metrics
            sensitivity, false_positive_rate, false_negative_rate, _ = self.calculate_metrics(ground_truth_labels, preds_classes)

            # Append to metrics summary for each event
            self.metrics_summary = self.metrics_summary.append({
                'CSV_File': csv_file,
                'eventId': eventIds[0] if eventIds else 'N/A',
                'userId': userIds[0] if userIds else 'N/A',
                'Accuracy': None,
                'Sensitivity': sensitivity,
                'False_Positive_Rate': false_positive_rate,
                'False_Negative_Rate': false_negative_rate,
                'False_Alarm_Rate': overall_false_alarm_rate
            }, ignore_index=True)

            # Store event metrics
            self.event_metrics.append({
                'eventId': eventIds[0] if eventIds else 'N/A',
                'userId': userIds[0] if userIds else 'N/A',
                'Sensitivity': sensitivity,
                'False_Positive_Rate': false_positive_rate,
                'False_Negative_Rate': false_negative_rate,
                'False_Alarm_Rate': overall_false_alarm_rate
            })

            # Update user-level metrics
            userId = userIds[0] if userIds else 'N/A'
            if userId != 'N/A':
                if userId not in self.user_metrics:
                    self.user_metrics[userId] = {'Sensitivity': 0, 'False_Positive_Rate': 0, 'False_Negative_Rate': 0, 'False_Alarm_Rate': 0, 'count': 0}

                self.user_metrics[userId]['Sensitivity'] += sensitivity
                self.user_metrics[userId]['False_Positive_Rate'] += false_positive_rate
                self.user_metrics[userId]['False_Negative_Rate'] += false_negative_rate
                self.user_metrics[userId]['False_Alarm_Rate'] += overall_false_alarm_rate
                self.user_metrics[userId]['count'] += 1

        # Save metrics summary
        self.metrics_summary.to_csv('AnalysedResults/event_metrics_summary.csv', index=False)

        # User summary
        user_summary_data = []
        for userId, metrics in self.user_metrics.items():
            if metrics['count'] > 0:
                user_summary_data.append({
                    'userId': userId,
                    'Average_Sensitivity': metrics['Sensitivity'] / metrics['count'],
                    'Average_False_Positive_Rate': metrics['False_Positive_Rate'] / metrics['count'],
                    'Average_False_Negative_Rate': metrics['False_Negative_Rate'] / metrics['count'],
                    'Average_False_Alarm_Rate': metrics['False_Alarm_Rate'] / metrics['count'],
                })

        user_summary_df = pd.DataFrame(user_summary_data)
        user_summary_df.to_csv('AnalysedResults/user_event_metrics_summary.csv', index=False)

        # Save event metrics summary
        event_metrics_df = pd.DataFrame(self.event_metrics)
        event_metrics_df.to_csv('AnalysedResults/event_level_metrics_summary.csv', index=False)

        print("Metrics summary saved.")
