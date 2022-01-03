from data_loader.load_tapping_data import TappingDataLoader


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def data_loader():
    # Get an object of Typing Data loader for loading Tapping data
    tapping_data = TappingDataLoader(username="ghaffh1@mcmaster.ca",
                                     password="As@hn6162")
    # Load the data
    temp_data = tapping_data.load_data(2)

    print(temp_data.head(2))

    # TODO Adding the query to get the number of records for all available data
    # TODO Loading the Data based on the files that to be loaded.
    # TODO Extract features from the files.
    # TODO Do simple feature fusion to creat the feature vector.
    # TODO normalize the features.
    # TODO Do outlier detection.
    # TODO Try different clustering algorithms.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_loader()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
