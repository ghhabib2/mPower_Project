from data_loader.data_loader import DataLoader
from utils import voice_feature_extractor
import os
import pandas as pd

class VoiceDataLoader(DataLoader):
    """ Load the Tapping data based on the Query """

    def load_data(self, limit):
        # Query mPoser Project
        table = self.syn.tableQuery(f"""
        SELECT  *
        FROM syn5511444
        LIMIT {limit}
        """)

        # Convert to the DataFrame
        df = table.asDataFrame()

        return df

    def feature_extractor(self):
        """
        Extract the features based on the list of the files pre-processed

        :return: Nothing
        :rtype: None
        """

        # Build the pass for the unique healthCode list
        file_path = os.path.join(self.ROOT_PATH, "unique_healthCode_list.csv")

        # Read the data of the unique healthCodes to the DataFrame
        health_code_df = pd.read_csv(file_path)

        # Iterate in healthCode records
        for _, health_code in health_code_df.iterrows():

            # Read the voice data belongs to the target healthCode
            print("Collecting voice information for ", health_code['healthCode'])
            query_builder = f"""SELECT *  
                                FROM syn5511444 
                                Where
                                    healthCode = '{health_code['healthCode']}'
                                """

            # Convert to the DataFrame
            voice_data = self.syn.tableQuery(query_builder)
            temp_voice_df = voice_data.asDataFrame()

            # Iterate in the list of the records
            for _, row in temp_voice_df.iterrows():
                # Read the name fo the file
                voice_file_path = os.path.join(self.VOICE_DATA_PATH, f"{row['audio_audio.m4a']}.m4a")
                # Check if the file exists
                if os.path.isfile(voice_file_path):
                    # Extract the features
                    print(f"Extracting features for {row['audio_audio.m4a']}")
                    segment_feature_name, segments_feature, segments_f0 = voice_feature_extractor.matlab_base_feature_downloader(voice_file_path)
                    # TODO Create folder for each file with the name of the file.
                    # TODO Store the extracted features in seperated npz file.
                else:
                    # Show alert that the file is not in the pre-processed data
                    print("The file removed in pre-process phase. We are deleting the countdown recording as well.")
        raise NotImplemented("This function has not been implemented yet")

