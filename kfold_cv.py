import numpy as np
from sklearn.model_selection import KFold
from config import config


class KFoldCrossValidation:
    def __init__(self, ts_model, X_train, y_train, batch_size=32, epochs=2, k=2):
        self.ts_model = ts_model
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.epochs = epochs
        self.k = k

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
            self.ts_model.train_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val, epochs=self.epochs, batch_size=self.batch_size)
            test_loss, test_accuracy = self.ts_model.evaluate_model(X_fold_val, y_fold_val)
            all_test_losses.append(test_loss)
            all_test_accuracies.append(test_accuracy)
            print(f"Fold {fold + 1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        avg_test_loss = np.mean(all_test_losses)
        avg_test_accuracy = np.mean(all_test_accuracies)
        print(f"Average Test Loss: {avg_test_loss:.4f}, Average Test Accuracy: {avg_test_accuracy:.4f}")
