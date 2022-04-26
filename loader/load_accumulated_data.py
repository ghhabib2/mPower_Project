import pandas as pd

from loader import DataLoader
# Import OS for IO
import os
from shutil import copyfile
import time
# Import for loading data
import pandas as np

user_home_path = user_path = os.path.expanduser("~")

ROOT_PATH = os.path.join(user_home_path, "Documents/collected_data_mpower")
MEMORY_DATA_PATH = os.path.join(ROOT_PATH, "memory_data")
WALKING_DATA = os.path.join(ROOT_PATH, "walking_data")
TPPING_DATA_PATH = os.path.join(ROOT_PATH, "tapping_data")
VOICE_DATA_PATH = os.path.join(ROOT_PATH, "voice_data")


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
        file_path = os.path.join(ROOT_PATH, "unique_healthCode_list.csv")

        # if os.path.exists(file_path) is not True:
        #     # Raise the error message and force user to invoke the correct method
        #     # For extracting unique healthCode IDs.
        #     print(f"""I could not find the file. Please double check and try to  extract the unique healthCode
        #     IDs before calling this function""")
        #     raise IOError

        # Read file into a Data frame
        health_code_df = pd.read_csv(file_path)

        # Tapping informatoin
        # tapping_df = None
        # Memory Game information
        # memory_df = None
        # Balance and gait information
        # walking_df = None
        # Voice information
        voice_df = None

        print("Start process")

        # Iterate records
        for _, healt_code in health_code_df.iterrows():
            # print("Processing for ", healt_code["healthCode"])
            # # Read the tapping information
            # # start with fetching the Tapping information
            # # ===========================================
            # print("Collecting tapping information for ", healt_code['healthCode'])
            # query_builder = f"""SELECT *
            #                     FROM syn5511439
            #                     Where
            #                         healthCode = '{healt_code['healthCode']}'
            #                     """
            # # Convert to the DataFrame
            # tapping_data = self.syn.tableQuery(query_builder)
            #
            # temp_tapping_df = tapping_data.asDataFrame()
            #
            # tapMap = self.syn.downloadTableColumns(tapping_data, ["tapping_results.json.TappingSamples"])
            #
            # for row in tapMap.items():
            #     # Target file address
            #     dist_path = os.path.join(TPPING_DATA_PATH, f"{row[0]}.json")
            #     src_path = row[1]
            #     # Copy file to the new path
            #     copyfile(src_path, dist_path)
            #
            # if tapping_df is None:
            #     tapping_df = temp_tapping_df
            # else:
            #     tapping_df = pd.concat((tapping_df, temp_tapping_df))

            # Read the memory information
            # start with fetching the memory information
            # ===========================================
            # print("Collecting memory information for ", healt_code['healthCode'])
            # query_builder = f"""SELECT *
            #                     FROM syn5511434
            #                     Where
            #                         healthCode = '{healt_code['healthCode']}'
            #                     """
            # # Convert to the DataFrame
            # memory_data = self.syn.tableQuery(query_builder)
            #
            # temp_memory_df = memory_data.asDataFrame()
            #
            # memoryMap = self.syn.downloadTableColumns(memory_data, ["MemoryGameResults.json.MemoryGameGameRecords"])
            #
            # for row in memoryMap.items():
            #     # Target file address
            #     dist_path = os.path.join(MEMORY_DATA_PATH, f"{row[0]}.json")
            #     src_path = row[1]
            #     # Copy file to the new path
            #     copyfile(src_path, dist_path)
            #
            # if memory_df is None:
            #     memory_df = temp_memory_df
            # else:
            #     memory_df = pd.concat((memory_df, temp_memory_df))
            #
            # # Read the motion information
            # # start with fetching the walking information
            # # ===========================================
            # print("Collecting walking information for ", healt_code['healthCode'])
            # query_builder = f"""SELECT *
            #                     FROM syn5511449
            #                     Where
            #                         healthCode = '{healt_code['healthCode']}'
            #                     """
            # # Convert to the DataFrame
            # walking_data = self.syn.tableQuery(query_builder)
            #
            # temp_walking_df = walking_data.asDataFrame()
            #
            # walkingMap = self.syn.downloadTableColumns(walking_data,
            #                                            ["accel_walking_outbound.json.items",
            #                                             "deviceMotion_walking_outbound.json.items",
            #                                             "pedometer_walking_outbound.json.items",
            #                                             "accel_walking_return.json.items",
            #                                             "deviceMotion_walking_return.json.items",
            #                                             "pedometer_walking_return.json.items",
            #                                             "accel_walking_rest.json.items",
            #                                             "deviceMotion_walking_rest.json.items"])
            #
            # for row in walkingMap.items():
            #     # Target file address
            #     dist_path = os.path.join(WALKING_DATA, f"{row[0]}.json")
            #     src_path = row[1]
            #     # Copy file to the new path
            #     copyfile(src_path, dist_path)
            #
            # if walking_df is None:
            #     walking_df = temp_walking_df
            # else:
            #     walking_df = pd.concat((walking_df, temp_walking_df))

            # Read the voice information
            # start with fetching the walking information
            # ===========================================

            print("Collecting voice information for ", healt_code['healthCode'])
            query_builder = f"""SELECT *  
                                       FROM syn5511444 
                                       Where
                                           healthCode = '{healt_code['healthCode']}'
                                       """
            # Convert to the DataFrame
            voice_data = self.syn.tableQuery(query_builder)

            temp_voice_df = voice_data.asDataFrame()

            voiceMap = self.syn.downloadTableColumns(voice_data,
                                                     ["audio_audio.m4a",
                                                      "audio_countdown.m4a",
                                                      ])

            for row in voiceMap.items():
                # Target file address
                dist_path = os.path.join(VOICE_DATA_PATH, f"{row[0]}.m4a")
                src_path = row[1]
                # Copy file to the new path
                copyfile(src_path, dist_path)
                # Delete the file in the src
                os.remove(src_path)
            if voice_df is None:
                voice_df = temp_voice_df
            else:
                voice_df = pd.concat((voice_df, temp_voice_df))

            print("Process ends for ", healt_code["healthCode"])
            print("===========================================")
            time.sleep(5)

        print("Export to the CSV")

        # Create tapping csv path
        # tapping_data_csv_path = os.path.join(ROOT_PATH, "tapping_unique_csv_data.csv")
        # memory_data_csv_path = os.path.join(ROOT_PATH, "memory_unique_csv_data.csv")
        # walking_data_csv_path = os.path.join(ROOT_PATH, "walking_unique_csv_data.csv")
        voice_data_csv_path = os.path.join(ROOT_PATH, "voice_unique_csv_data.csv")

        # Save data to CSV
        # walking_df.to_csv(walking_data_csv_path, index=False)
        # memory_df.to_csv(memory_data_csv_path, index=False)
        # tapping_df.to_csv(tapping_data_csv_path, index=False)
        voice_df.to_csv(voice_data_csv_path, index=False)

        print("Done.")
