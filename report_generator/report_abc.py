import os
import abc

# Root of the system
user_home_path = os.path.expanduser("~")


class ReportABCCLass(object):
    """
    Abstract class that represent the interface of the Model trainer class
    """

    def __init__(self, to_read_dir_path):
        """

        :param (str) to_read_dir_path: The directory path for reading the data
        """
        self._ROOT_PATH = os.path.join(user_home_path, "Documents/collected_data_mpower")
        self._TO_READ_PATH = os.path.join(self._ROOT_PATH, to_read_dir_path)

    @abc.abstractmethod
    def load(self):
        """
        Load the data to be used for training and validation/testing

        :return: Return the extracted data
        """

        raise NotImplemented("This method has not implemented yet.")

    @abc.abstractmethod
    def plot(self):
        """
        Performing the training

        :return: Return the trained model
        """

        raise NotImplemented("This method has not implemented yet.")
