# library for downloading the data from synapse service
import synapseclient
# Pandas library for keeping the information of the table
import pandas as pd
# Library for Abstract Class definition
import abc


class DataLoader(object):
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

        try:
            self.syn = synapseclient.login(username, password)
        except Exception as ex:
            raise ConnectionError("""
            Connection error to synapse service. Please check your connection or your 
            login information or try a few minutes later. Here is the Exception message: """, str(ex)
                                  )

    @abc.abstractmethod
    def load_data(self, limit):
        """
        Load data as a DataFrame include all the information downloaded from the synapse dataset

        :param limit: Number of records to be downloaded.
        :type limit: int
        :return: Return the Data Frame with table data based on the limitation
        :rtype: pd.DataFrame
        """
        raise NotImplemented("The invoked method has not implemented yet")
