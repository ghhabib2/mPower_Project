import pandas as pd


from data_loader.data_loader import DataLoader
# Import OS for IO
import os
# Import for loading data
import pandas as np

ROOT_URL = os.path.join(os.getcwd(), "collected_data")


class AccumulatedDataLoader(DataLoader):
    """
    Accumulated data collection process.
    """

    def load_unique_data_records(self):
        """
        Load the unique data records based on the uniquely extracted healthCodes

        :return: Data Frame with all necessary information
        :rtype: pd.DataFrame
        """

        # load the data based on the previously saved unique data records.
        # Check for existence of the file:
        file_path = os.path.join(ROOT_URL, "unique_healthCode_list.csv")

        if os.path.exists(file_path) is not True:
            # Raise the error message and force user to invoke the correct method
            # For extracting unique healthCode IDs.
            print(f"""I could not find the file. Please double check and try to  extract the unique healthCode
            IDs before calling this function""")
            raise IOError

        # Read file into a Data frame
        health_code_df = pd.read_csv(file_path)

        # Iterate records
        for _, healt_code in health_code_df.iterrows():
            # Read the tapping information
            # Query for checking the information of the tapping
            query_builder = f"""SELECT *  
                                FROM syn5511439 
                                Where
                                    healthCode = '{healt_code['healthCode']}'
                                """
            # Convert to the DataFrame
            tapping_data = self.syn.tableQuery(query_builder)
            tapMap = self.syn.downloadTableColumns(tapping_data,
                                                   ["accel_tapping.json.items", "tapping_results.json.TappingSamples"])
