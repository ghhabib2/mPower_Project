from data_loader import AccumulatedDataLoader


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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # data_loader()
    data_loader()
