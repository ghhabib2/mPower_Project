# library for downloading the data from synapse service
import synapseclient
from datetime import datetime, timedelta
# Pandas library for keeping the information of the table
import pandas as pd
# Import reduce function to use in merge operation
from functools import reduce
# Library for Abstract Class definition
import abc
# Adding the OS for file management
import os


ROOT = os.path.join(os.getcwd(), "collected_data")


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

    @property
    def unique_data_record_number(self):
        """
        :return: Return the number of unique data record number
        :rtype: int
        """

        # Query for checking the information of the walking
        query_builder = '''
        SELECT DISTINCT healthCode
        FROM syn5511449
        '''
        # Convert to the DataFrame and delete the duplicates
        walking_df = self.syn.tableQuery(query_builder).asDataFrame().drop_duplicates()
        i = 0
        for _, row in walking_df.iterrows():

            # creating a dataframe for walking for
            temp_walking_df = pd.DataFrame(data=[[row['healthCode']]],
                                           columns=['healthCode'])

            # Query for checking the information of the voice
            query_builder = f"SELECT count(healthCode) FROM syn5511444 Where healthCode ='{row['healthCode']}'"
            # Convert to the DataFrame
            voice_data_record_numbers = self.syn.tableQuery(query_builder).asInteger()

            # Query for checking the information of the tapping
            query_builder = f"SELECT count(healthCode) FROM syn5511439 Where healthCode = '{row['healthCode']}'"
            # Convert to the DataFrame
            tapping_data_record_numbers = self.syn.tableQuery(query_builder).asInteger()

            # Query for checking the information of the memory
            query_builder = f"SELECT count(healthCode) FROM syn5511434 Where healthCode = '{row['healthCode']}'"
            # Convert to the DataFrame
            memory_data_record_numbers = self.syn.tableQuery(query_builder).asInteger()

            if voice_data_record_numbers > 0 and tapping_data_record_numbers > 0 and memory_data_record_numbers > 0:
                # Add the sub list to the over all lists
                try:
                    merge_list = pd.concat([merge_list, temp_walking_df], axis=0).drop_duplicates()

                except NameError:
                    merge_list = temp_walking_df

                i += 1
                print(f"{i} - I've added the following healthCode to the list: {row['healthCode']}")

        # Adding the files to a CSV file
        # ==============================

        # Create the Directory if it is not exist
        if os.path.isdir(ROOT) is not True:
            os.mkdir(ROOT)

        # file that going to keep the unique files' data
        file_path = os.path.join(ROOT, "unique_healthCode_list.csv")

        print("I am about to save the unique health codes data in the csv file")

        # Store the files in the csv file
        merge_list.to_csv(file_path, index=False)

        print("I am done.")

        # Return the number of records in dataframe
        return len(merge_list)
