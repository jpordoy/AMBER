class config:

    #Globals
    batch_size = 8
    num_classes = 3  # classes, seizure/no seizure
    epochs = 25      # Epoch iterations
    row_hidden = 64  # hidden neurons in conv layers
    col_hidden = 64   # hidden neurons in the Bi-LSTM layers
    RANDOM_SEED = 123    
    N_TIME_STEPS = 125   # 50 records in each sequence
    N_FEATURES = 2      # mag,hr,roi_Ratio,output
    step = 100           # window overlap = 50 -10 = 40  (80% overlap)
    N_CLASSES = 3       # class labels