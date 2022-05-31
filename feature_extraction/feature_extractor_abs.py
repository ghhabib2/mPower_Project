import os
import abc

# Root of the system
user_home_path = user_path = os.path.expanduser("~")


class FeatureExtractor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, to_read_dir_path, to_store_dir_path):
        """

        :param (str) to_read_dir_path: Directory path to read the files
        :param (str) to_store_dir_path: Directory path to store the features
        """
        # Keep the path for root folder of the data`
        self.ROOT_PATH = os.path.join(user_home_path, "Documents/collected_data_mpower")
        self.TO_READ_PATH = os.path.join(self.ROOT_PATH, to_read_dir_path)
        self.TO_STORE_PATH = os.path.join(self.ROOT_PATH,to_store_dir_path)

    @abc.abstractmethod
    def loader(self, file_path):
        """
        Function Loader is responsible for loading a file

        :param (str) file_path: File path for the file to be loaded
        :return: Result of the load operation
        :raises IOError: It might be possible that the system could not read the file
        """

        raise NotImplemented("This method has not implemented yet.")

    @abc.abstractmethod
    def process(self):
        """
        Process all the files in the folder

        :return: Return Nothing
        :rtype: None
        :raises IOError, OSError, RuntimeError:
        """

        raise NotImplemented("This method has not implemented yet.")

    @abc.abstractmethod
    def save_feature(self, feature, file_path):
        """
        Save the feature based on the file path
        """
        raise NotImplemented("This method has not implemented yet.")