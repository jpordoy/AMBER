
import matplotlib.pyplot as plt
import os
import seaborn as sns

class KFoldCrossValidation:
    def __init__(self, ts_model, X_train, y_train, batch_size=32, epochs=25, k=5, save_dir='plots'):
        self.ts_model = ts_model
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.epochs = epochs
        self.k = k
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)  # Create directory if it doesn't exist
        self.history_accumulated = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}  # Initialize empty dictionaries to accumulate metrics
        self.fold_history = []  # Initialize list to store individual fold histories
        
    def plot_confusion_matrix(self, fold, confusion_mat):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - Fold {fold + 1}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(os.path.join(self.save_dir, f'confusion_matrix_fold_{fold + 1}.png'))  # Save each plot with a unique filename
        plt.close()
        
    def plot_individual_metrics(self, fold):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Create subplots for accuracy and loss

        # Plot training accuracy
        axs[0].plot(self.fold_history[fold]['accuracy'], label='Training Accuracy')
        axs[0].plot(self.fold_history[fold]['val_accuracy'], label='Validation Accuracy')
        axs[0].set_title('Accuracy - Fold {}'.format(fold + 1))
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Accuracy')
        axs[0].legend()

        # Plot training loss
        axs[1].plot(self.fold_history[fold]['loss'], label='Training Loss')
        axs[1].plot(self.fold_history[fold]['val_loss'], label='Validation Loss')
        axs[1].set_title('Loss - Fold {}'.format(fold + 1))
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'individual_metrics_fold_{fold + 1}.png'))  # Save the plot with a unique filename
        plt.close()  # Close the plot to avoid displaying it

    def plot_overall_metrics(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create subplots for accuracy, validation accuracy, loss, and validation loss

        # Plot overall accuracy
        for fold in range(self.k):
            axs[0, 0].plot(range(1, self.epochs + 1), self.fold_history[fold]['accuracy'], label=f'Fold {fold + 1}')
            axs[0, 0].set_title('Accuracy')
            axs[0, 0].set_xlabel('Epoch')
            axs[0, 0].set_ylabel('Accuracy')
            axs[0, 0].legend()

        # Plot overall validation accuracy
        for fold in range(self.k):
            axs[0, 1].plot(range(1, self.epochs + 1), self.fold_history[fold]['val_accuracy'], label=f'Fold {fold + 1}')
            axs[0, 1].set_title('Validation Accuracy')
            axs[0, 1].set_xlabel('Epoch')
            axs[0, 1].set_ylabel('Accuracy')
            axs[0, 1].legend()

        # Plot overall loss
        for fold in range(self.k):
            axs[1, 0].plot(range(1, self.epochs + 1), self.fold_history[fold]['loss'], label=f'Fold {fold + 1}')
            axs[1, 0].set_title('Loss')
            axs[1, 0].set_xlabel('Epoch')
            axs[1, 0].set_ylabel('Loss')
            axs[1, 0].legend()

        # Plot overall validation loss
        for fold in range(self.k):
            axs[1, 1].plot(range(1, self.epochs + 1), self.fold_history[fold]['val_loss'], label=f'Fold {fold + 1}')
            axs[1, 1].set_title('Validation Loss')
            axs[1, 1].set_xlabel('Epoch')
            axs[1, 1].set_ylabel('Loss')
            axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'overall_metrics.png'))
        plt.close()  # Close the plot to avoid displaying it

    def run(self):
        kf = KFold(n_splits=self.k, shuffle=True)
        all_test_losses = []
        all_test_accuracies = []
        for fold, (train_index, test_index) in enumerate(kf.split(self.X_train[0])):
            print(f"Fold {fold + 1}/{self.k}")
            X_fold_train = [X[train_index] for X in self.X_train]
            y_fold_train = self.y_train[train_index]
            X_fold_val = [X[test_index] for X in self.X_train]
            y_fold_val = self.y_train[test_index]
            self.ts_model.build_model(num_features=2, input_shape=(config.N_TIME_STEPS, 1))
            self.ts_model.compile_model()
            history = self.ts_model.train_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val, epochs=self.epochs, batch_size=self.batch_size)
            test_loss, test_accuracy = self.ts_model.evaluate_model(X_fold_val, y_fold_val)
            all_test_losses.append(test_loss)
            all_test_accuracies.append(test_accuracy)
            print(f"Fold {fold + 1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            
            # Accumulate history
            self.fold_history.append(history.history)
            
            # Generate confusion matrix
            y_pred_val = self.ts_model.predict(X_fold_val)
            y_pred_classes = np.argmax(y_pred_val, axis=1)
            y_true_classes = np.argmax(y_fold_val, axis=1)
            
            confusion_mat = confusion_matrix(y_true_classes, y_pred_classes)

            # Print confusion matrix
            print(f"Confusion Matrix for Fold {fold + 1}:\n{confusion_mat}\n")

            # Save confusion matrix plot
            self.plot_confusion_matrix(fold, confusion_mat)
        
            # Save individual plots
            self.plot_individual_metrics(fold)
        
        avg_test_loss = np.mean(all_test_losses)
        avg_test_accuracy = np.mean(all_test_accuracies)
        print(f"Average Test Loss: {avg_test_loss:.4f}, Average Test Accuracy: {avg_test_accuracy:.4f}")
        
        # Plot overall metrics
        self.plot_overall_metrics()

        return self.history_accumulated
