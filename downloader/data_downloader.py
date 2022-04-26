# library for downloading the data from synapse service
import synapseclient
# Library for Abstract Class definition
import abc
# Adding the OS for file management
import os

user_home_path = user_path = os.path.expanduser("~")


class DataDownloader(object):
    """
    Abstract Class for data loading
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, username, password):
        """
        Initialize the object
        :param username: Username for Synapse Service
        :type username: str
        :param password: Password of the user for Synapse Service
        :type password: str
        """

        self.ROOT_PATH = os.path.join(user_home_path, "Documents/collected_data_mpower")
        self.MEMORY_DATA_PATH = os.path.join(self.ROOT_PATH, "memory_data")
        self.WALKING_DATA = os.path.join(self.ROOT_PATH, "walking_data")
        self.TAPPING_DATA_PATH = os.path.join(self.ROOT_PATH, "tapping_data")
        self.VOICE_DATA_PATH = os.path.join(self.ROOT_PATH, "voice_data")

        try:
            self.syn = synapseclient.login(username, password)
        except Exception as ex:
            raise ConnectionError("""
            Connection error to synapse service. Please check your connection or your 
            login information or try a few minutes later. Here is the Exception message: """, str(ex)
                                  )

    @abc.abstractmethod
    def download_data(self, csv_file_path):
        """
        Download Data based on the type of task

        :param csv_file_path : CSV file path that keeps the data records information
        :type csv_file_path : str
        :return: Return True if the target dota downloaded and false in else case
        :rtype: bool
        """
        raise NotImplemented("The invoked method has not implemented yet")
