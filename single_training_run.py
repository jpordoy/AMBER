from .config import config


class SingleTrainingRun:
    def __init__(self, ts_model, X_train, y_train, X_val, y_val, batch_size=32, epochs=2):
        self.ts_model = ts_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.epochs = epochs

    def run(self):
        print("Training the model on the entire training dataset")
        # Build and compile the model
        self.ts_model.build_model(num_features=2, input_shape=(config.N_TIME_STEPS, 1))
        self.ts_model.compile_model()
        # Train the model
        self.ts_model.train_model(self.X_train, self.y_train, self.X_val, self.y_val, epochs=self.epochs, batch_size=self.batch_size)
        # Evaluate the model on the validation set
        test_loss, test_accuracy = self.ts_model.evaluate_model(self.X_val, self.y_val)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
