import csv

import numpy as np

from loader import DataLoader
from utils import voice_feature_extractor
import os
import pandas as pd
import gc


class VoiceDataLoader(DataLoader):
    """ Load the Tapping data based on the Query """

    def load_data(self, limit=None):
        labels = ["take Parkinson medications",
                  "Immediately before Parkinson medication",
                  "Just after Parkinson medication (at your best)",
                  "Another time"]
        # Initiate the output DataFrame as a None varialbe
        df = None

        table = self.syn.tableQuery("SELECT  * FROM syn5511444")
        # Check if the output DataFrame is empty
        df = table.asDataFrame()

        # Set the Labels
        for index, row in df.iterrows():
            if row['medTimepoint'] == "I don't take Parkinson medications":
                df.at[index, 'medTimepoint'] = 0
            elif row['medTimepoint'] == "Immediately before Parkinson medication":
                df.at[index, 'medTimepoint'] = 1
            elif row['medTimepoint'] == "Just after Parkinson medication (at your best)":
                df.at[index, 'medTimepoint'] = 2
            elif row['medTimepoint'] == "Another time":
                df.at[index, 'medTimepoint'] = 3

        # Return the final result
        if limit is None:
            return df
        else:
            # Label is equal to 0
            output_df = df[df['medTimepoint'] == 0].head(limit)
            # Label is equal to 1
            output_df = pd.concat([output_df, df[df['medTimepoint'] == 1].head(limit)])
            # Label is equal to 2
            output_df = pd.concat([output_df, df[df['medTimepoint'] == 2].head(limit)])
            # Label is equal to 3
            output_df = pd.concat([output_df, df[df['medTimepoint'] == 3].head(limit)])
            return output_df

    def voice_feature_extractor_matlab(self, data_file_path, data_folder_path, voice_folder_path):

        # Build the pass for csv file
        file_path = os.path.join(self.ROOT_PATH, data_file_path)
        voice_directory_path = os.path.join(self.ROOT_PATH, voice_folder_path)
        directory_path = os.path.join(self.ROOT_PATH, data_folder_path)


        if not os.path.exists(directory_path):
            print("The directory is not exists.")
            os.mkdir(directory_path)
            print("The directory has been created.")

        voice_feature_list = os.listdir(directory_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError("The CSV file is not exists. Check the file path")

        # Load the csv file into a pandas DataFrame
        voices_data_frame = pd.read_csv(file_path)

        # Iterate over the rows of the pandas DataFrame to download each audio_countdown
        i = 1
        row_collection = []
        for _, row in voices_data_frame.iterrows():

            voice_file_path = os.path.join(voice_directory_path, f"{row['audio_audio.m4a']}.m4a")

            # Check if the count_down version of the audio file exist
            if not os.path.exists(voice_file_path):
                # The file has been cleared because of the background noise. Print the message for not downloading
                # and continue.
                print(f"File dose not exist. Feature cannot be extracted for {row['recordId']}")
                continue

            # Extract the features
            # Check if the folder name is in the list
            if not any(str(row['audio_audio.m4a']) in s for s in voice_feature_list):
                # Start feature extraction process
                # Extract the features
                print(f"{i}- Extracting features for {row['audio_audio.m4a']}")
                segment_feature_name, segments_feature, segments_f0 = voice_feature_extractor \
                    .matlab_base_feature_downloader(voice_file_path, segment_duration=1)

                if segment_feature_name is None:
                    print("Feature extraction encountered a problem. File skipped.")
                    continue

                # Create a folder for storing the voice features for a target file
                features_folder = os.path.join(directory_path, str(row['audio_audio.m4a']))

                # Save the list of the features
                feature_name_path = os.path.join(directory_path, "features_name.txt")

                if not os.path.isfile(feature_name_path):
                    with open(feature_name_path, 'w') as f:
                        for line in segment_feature_name.tolist():
                            f.write(line)
                            f.write("\n")

                if not os.path.isdir(features_folder):
                    os.mkdir(features_folder)

                # Iterate in extracted features
                for index, feature_vector in enumerate(segments_feature):
                    segment_feature_file_path = os.path.join(features_folder,
                                                             f"{row['audio_audio.m4a']}_s_f_{index}.npz")
                    f0_feature_file_path = os.path.join(features_folder,
                                                        f"{row['audio_audio.m4a']}_s_f_f0{index}.npz")

                    # Add the features
                    np.savez(segment_feature_file_path, s_f=feature_vector)
                    np.savez(f0_feature_file_path, f0_f=segments_f0[index])

                print(f"Features extracted for the the voice file {row['audio_audio.m4a']}")
                # Add the folder to list of the files features extracted for them
                voice_feature_list.append(str(row['audio_audio.m4a']))
                row_collection.append([row['healthCode'],
                                       row['audio_audio.m4a'],
                                       row["medTimepoint"],
                                       row['createdOn']])
                i += 1

            else:
                # Print a message notifiying the features already extracted.
                print(f"{i}- The features already extracted for this file.")
                row_collection.append([row['healthCode'],
                                       row['audio_audio.m4a'],
                                       row["medTimepoint"],
                                       row['createdOn']])
                i += 1

        # Add the rows to feature collection file
        # creat a temp file for storing the features basic information.
        temp_dataset_df = pd.DataFrame(data=np.array(row_collection), columns=['healthCode',
                                                                               'audio_audio',
                                                                               'medTimepoint',
                                                                               'createdOn'])

        # generate the path for the file
        csv_file_path = os.path.join(self.ROOT_PATH, "voice_feature_extraction_no_limit.csv")

        # Store the file
        temp_dataset_df.to_csv(csv_file_path, index=False)

        print("The features extracted successfully.")

    def voice_feature_extractor_praa(self, data_file_path, data_folder_path, voice_folder_path):

        # Build the pass for csv file
        file_path = os.path.join(self.ROOT_PATH, data_file_path)
        voice_directory_path = os.path.join(self.ROOT_PATH, voice_folder_path)
        directory_path = os.path.join(self.ROOT_PATH, data_folder_path)


        if not os.path.exists(directory_path):
            print("The directory is not exists.")
            os.mkdir(directory_path)
            print("The directory has been created.")

        voice_feature_list = os.listdir(directory_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError("The CSV file is not exists. Check the file path")

        # Load the csv file into a pandas DataFrame
        voices_data_frame = pd.read_csv(file_path)

        # Iterate over the rows of the pandas DataFrame to download each audio_countdown
        i = 1
        row_collection = []
        for _, row in voices_data_frame.iterrows():

            voice_file_path = os.path.join(voice_directory_path, f"{row['audio_audio.m4a']}.m4a")

            # Check if the count_down version of the audio file exist
            if not os.path.exists(voice_file_path):
                # The file has been cleared because of the background noise. Print the message for not downloading
                # and continue.
                print(f"File dose not exist. Feature cannot be extracted for {row['recordId']}")
                continue

            # Extract the features
            # Check if the folder name is in the list
            if not any(str(row['audio_audio.m4a']) in s for s in voice_feature_list):
                # Start feature extraction process
                # Extract the features
                print(f"{i}- Extracting features for {row['audio_audio.m4a']}")
                segments_feature = voice_feature_extractor.praa_base_feature_downloader(voice_file_path,
                                                                                        segment_duration=1)

                if segments_feature is None:
                    print(f"Features not extracted Proeprly. Data record {row['audio_audio.m4a']} has been removed.")
                    continue

                # Create a folder for storing the voice features for a target file
                features_folder = os.path.join(directory_path, str(row['audio_audio.m4a']))

                if not os.path.isdir(features_folder):
                    os.mkdir(features_folder)

                # Iterate in extracted features
                for index, feature_vector in enumerate(segments_feature):
                    segment_feature_file_path = os.path.join(features_folder,
                                                             f"{row['audio_audio.m4a']}_s_f_{index}.npz")

                    # Add the features
                    np.savez(segment_feature_file_path, s_f=feature_vector)

                print(f"Features extracted for the the voice file {row['audio_audio.m4a']}")
                # Add the folder to list of the files features extracted for them
                voice_feature_list.append(str(row['audio_audio.m4a']))
                row_collection.append([row['healthCode'],
                                       row['audio_audio.m4a'],
                                       row["medTimepoint"],
                                       row['createdOn']])
                i += 1

            else:
                # Print a message notifiying the features already extracted.
                print(f"{i}- The features already extracted for this file.")
                row_collection.append([row['healthCode'],
                                       row['audio_audio.m4a'],
                                       row["medTimepoint"],
                                       row['createdOn']])
                i += 1

            gc.collect()

        # Add the rows to feature collection file
        # creat a temp file for storing the features basic information.
        temp_dataset_df = pd.DataFrame(data=np.array(row_collection), columns=['healthCode',
                                                                               'audio_audio',
                                                                               'medTimepoint',
                                                                               'createdOn'])

        # generate the path for the file
        csv_file_path = os.path.join(self.ROOT_PATH, "voice_feature_extraction_no_limit.csv")

        # Store the file
        temp_dataset_df.to_csv(csv_file_path, index=False)

        print("The features extracted successfully.")

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

        # Create a directory for holding the audio feature extracted
        voice_feature_folder_path = os.path.join(self.ROOT_PATH, "voice_features")

        if not os.path.isdir(voice_feature_folder_path):
            os.mkdir(voice_feature_folder_path)

        voice_feature_list = os.listdir(voice_feature_folder_path)

        feature_file_data_holder = []

        record_counter = 0
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
                record_counter += 1
                # Read the name fo the file
                voice_file_path = os.path.join(self.VOICE_DATA_PATH, f"{row['audio_audio.m4a']}.m4a")
                # Check if the file exists
                if os.path.isfile(voice_file_path):

                    # Check if the folder name is in the list
                    if not any(str(row['audio_audio.m4a']) in s for s in voice_feature_list):
                        # The feature is not in the list. Extract the features

                        # Extract the features
                        print(f"{record_counter}- Extracting features for {row['audio_audio.m4a']}")
                        segment_feature_name, segments_feature, segments_f0 = voice_feature_extractor \
                            .matlab_base_feature_downloader(voice_file_path)

                        if segment_feature_name is None:
                            print("Feature extraction encountered a problem. File skipped.")
                            continue

                        # Create a folder for storing the voice features for a target file
                        features_folder = os.path.join(voice_feature_folder_path, str(row['audio_audio.m4a']))

                        # Save the list of the features
                        feature_name_path = os.path.join(self.ROOT_PATH, "voice_features/features_name.txt")

                        if not os.path.isfile(feature_name_path):
                            with open(feature_name_path, 'w') as f:
                                for line in segment_feature_name.tolist():
                                    f.write(line)
                                    f.write("\n")

                        if not os.path.isdir(features_folder):
                            os.mkdir(features_folder)

                        # Iterate in extracted features
                        for index, feature_vector in enumerate(segments_feature):
                            segment_feature_file_path = os.path.join(features_folder,
                                                                     f"{row['audio_audio.m4a']}_s_f_{index}.npz")
                            f0_feature_file_path = os.path.join(features_folder,
                                                                f"{row['audio_audio.m4a']}_s_f_f0{index}.npz")

                            # Add the features
                            np.savez(segment_feature_file_path, s_f=feature_vector)
                            np.savez(f0_feature_file_path, f0_f=segments_f0[index])

                        print(f"Features extracted for the the voice file {row['audio_audio.m4a']}")
                        # Add the folder to list of the files features extracted for them
                        voice_feature_list.append(str(row['audio_audio.m4a']))

                        gc.collect()

                    else:
                        # Print a message notifiying the features already extracted.
                        print(f"{record_counter}- The features already extracted for this file.")

                else:
                    # Show alert that the file is not in the pre-processed data
                    print("The file removed in pre-process phase. We are deleting the countdown recording as well.")
                    # Remove the audio_countdown file.
                    try:
                        os.remove(os.path.join(self.VOICE_DATA_PATH, f"{row['audio_countdown.m4a']}.m4a"))
                    except IOError as ex:
                        print(f"The file already deleted. Here is the error: {str(ex)}")

                feature_file_data_holder.append([row['healthCode'],
                                                 row['audio_audio.m4a'],
                                                 row["medTimepoint"],
                                                 row['createdOn']])

        # creat a temp file for storing the features basic information.
        temp_dataset_df = pd.DataFrame(data=np.array(feature_file_data_holder), columns=['healthCode',
                                                                                         'audio_audio',
                                                                                         'medTimepoint',
                                                                                         'createdOn'])

        # generate the path for the file
        csv_file_path = os.path.join(self.ROOT_PATH, "voice_feature_extraction.csv")

        # Store the file
        temp_dataset_df.to_csv(csv_file_path, index=False)

        print("The features extracted successfully.")
