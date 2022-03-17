from data_loader import AccumulatedDataLoader
from featureExtraction import balanceHelper,gaitHelpers, tappingHelpers, memoryHelpers
from utils import signal_plot
import os
import numpy as np

ROOT_PATH = os.path.join(os.getcwd(), "collected_data")

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def data_loader():
    # Get an object of Typing Data loader for loading Tapping data
    data_loader_object = AccumulatedDataLoader(
        username='ghaffh1@mcmaster.ca',
        password='As@hn6162')

    data = data_loader_object.load_unique_data_records()

    # Print the number of data records
    print(f"Number of unique healthCode in the data set is: {data_loader_object.unique_data_record_number}")

    # TODO Adding the query to get the number of records for all available data
    # TODO Loading the Data based on the files that to be loaded.
    # TODO Extract features from the files.
    # TODO Do simple feature fusion to creat the feature vector.
    # TODO normalize the features.
    # TODO Do outlier detection.
    # TODO Try different clustering algorithms.

def plotter():
    balance_file_path = os.path.join(ROOT_PATH,"data_moition_signal_sample.csv")
    balance_file_distance_path = os.path.join(ROOT_PATH, "data_moition_signal_sample_distance.csv")
    tapinter_file_path = os.path.join(ROOT_PATH, "tap_inter.csv")

    balance_axis_file = np.genfromtxt(balance_file_path, delimiter=",")
    balance_distance_file = np.genfromtxt(balance_file_distance_path, delimiter=",")
    tapinter_file = np.genfromtxt(tapinter_file_path, delimiter=",")

    balance_time = balance_axis_file[:, 0]
    balance_x = balance_axis_file[:, 1]
    balance_y = balance_axis_file[:, 2]
    balance_z = balance_axis_file[:, 3]
    balance_distance = balance_distance_file

    signal_plot.signal_plotter(balance_time, balance_x, "Balance signal X axis")
    signal_plot.signal_plotter(balance_time, balance_y, "Balance signal Y axis")
    signal_plot.signal_plotter(balance_time, balance_z, "Balance signal Z axis")
    signal_plot.signal_plotter(balance_time, balance_distance, "Balance Distance from origin signal")
    signal_plot.array_plot(tapinter_file, "Tapping Interval")





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_loader()
    # balanceHelper.tested_jason()
    #  tappingHelpers.tested_jason()
    #memoryHelpers.tested_jason()
    #plotter()

