from AMBER.IctalPhaseDetectionProcessor.ipd_data_processor import IpdDataProcessor
from AMBER.OsdbDataProcessor.osdb_data_loader import OsdbDataLoader
from AMBER.config import Config


if __name__ == "__main__":
    
    # Example usage
    file_path = '../Data/osdb_3min_allSeizures.json'  # Replace with your JSON file path
    processor = OsdbDataLoader(file_path)
    df_result = processor.process_data()   
     
    labels = '../Data/ipd_labels.csv'  # Replace with your JSON file path

    #Select the eventIds required from the IPD labels dataset
    ids = [
        5635, 5637, 6668, 8726, 8738, 15923, 7219, 28725, 7222, 15417, 21561, 6717, 
        21569, 5705, 6732, 5721, 7258, 7772, 7262, 7775, 8800, 9828, 41062, 6761,
        44137, 6767, 5745, 115, 119, 5254, 7823, 6808, 45209, 6815, 42147, 5288,
        12973, 9401, 31420, 7357, 31421, 15039, 21695, 7365, 45781, 6884, 6897,
        9470, 8960, 5889, 5891, 9475, 7434, 8970, 21797, 8998, 9005, 4924, 24380,
        11587, 11591, 12618, 6476, 14157, 14159, 12624, 40784, 45393, 6998, 7006,
        7007, 15208, 21865, 21866, 5483, 21867, 26988, 5486, 26992, 7036, 15230,
        21886, 7044, 407, 47000, 47002, 9627, 53665, 53666, 5031, 12206, 12214,
        6587, 6590, 36799, 34756, 34759, 5580, 36812, 40913, 21458, 7125, 7126,
        8661, 26071, 5595, 5596, 12763, 26077, 5087, 5610, 6847
    ]    
    
     
    # Create an instance of the class
    processor = IpdDataProcessor(labels, ids)
    
    # Filter and sort the sensor data
    processor.filter_and_sort_sensor_data(df_result)
    # Merge the labels
    processor.merge_labels()
    
    # Get the final processed sensor data
    df_sensor_data = processor.get_sensor_data()
    
    print(df_sensor_data.tail())
    
    