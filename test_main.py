from test_seizure_predictior import SeizurePredictior

if __name__ == "__main__":
    # Define the priority event IDs
    priority_test_event_ids = [
        5595, 5596, 28725, 28734, 40913, 14101, 15208, 26071, 26077, 26988,
        21603, 21695, 21797, 21855, 15039, 12618, 12624, 12763, 5635, 5637, 6668, 8726,
        7219, 7222, 6732, 5721, 7258, 7262, 6761, 5745, 5254, 7823, 11591, 40784, 5610
    ]

    model = SeizurePredictior(file_path='Data/osdb_3min_allSeizures.json', priority_test_event_ids=priority_test_event_ids)
    y_pred = model.train_and_evaluate()
    model.plot_metrics(y_pred)


