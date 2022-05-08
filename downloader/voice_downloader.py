from downloader import DataDownloader

import pandas as pd
from shutil import copyfile
import os


class VoiceDownloader(DataDownloader):
    def download_data(self, csv_file_path):
        # Check if the file exists

        if not os.path.isfile(csv_file_path):
            raise FileNotFoundError("The CSV file is not exists. Check the file path")
        # Load the csv file into a pandas Data Frame
        # data_file = pd.read_csv(csv_file_path)

        raise NotImplemented("This method has not completely implemented.")

    def audio_countdown_downloader(self, csv_file_path, path):
        """
        Download the audio_countdown and store them in the dedicated path

        :param csv_file_path: CSV file path to be used for data downloading process
        :type csv_file_path: str
        :param: Path of the directory that audio file going to be stored in
        :type path : str
        :return: True if the download process is successful and False if it is not.
        :rtype: bool
        """

        try:
            # Generate the directory based on the root path
            directory_path = os.path.join(self.ROOT_PATH, path)

            if not os.path.exists(directory_path):
                print("The directory is not exists.")
                os.mkdir(directory_path)
                print("The directory has been created.")

            audio_file_path = os.path.join(self.ROOT_PATH, csv_file_path)

            if not os.path.isfile(audio_file_path):
                raise FileNotFoundError("The CSV file is not exists. Check the file path")

            # Load the csv file into a pandas DataFrame
            voices_data_frame = pd.read_csv(audio_file_path)

            # Iterate over the rows of the pandas DataFrame to download each audio_countdown
            i = 0
            for _, row in voices_data_frame.iterrows():
                i += 1
                # Read the data record from the repository
                query_builder = f"SELECT *  FROM syn5511444 Where recordId = '{row['recordId']}'"

                # Load the data
                voice_data = self.syn.tableQuery(query_builder)

                voice_data_df = voice_data.asDataFrame()

                dist_path = os.path.join(directory_path, f"{voice_data_df.iloc[0]['audio_countdown.m4a']}.m4a")

                if os.path.exists(dist_path):
                    print(f"Count down audio file - {voice_data_df.iloc[0]['audio_countdown.m4a']}.m4a - has already "
                          f"been downloaded. "
                          f"File count is {i}")
                    continue

                # map the files needed to be downloaded
                voice_map = self.syn.downloadTableColumns(voice_data, ["audio_countdown.m4a"])

                for record in voice_map.items():
                    # Generate the target file path
                    dist_path = os.path.join(directory_path, f"{record[0]}.m4a")
                    # Src file path
                    src_path = record[1]
                    # Copy file to the new path
                    copyfile(src_path, dist_path)
                    # Delete the file in the src
                    os.remove(src_path)
                    # Print a comment about downloading the file
                    print(f"Count down audio file - {record[0]}.m4a - downloaded. File count is {i}")
                    break

            # Process is successful
            return True
        except Exception as ex:
            print(f"A runtime error reported. Error:{str(ex)}")
            # Process failed
            return False

    def audio_downloader(self, csv_file_path, path, count_down_path):
        """
        Download the audio_countdown and store them in the dedicated path

        :param csv_file_path: CSV file path to be used for data downloading process
        :type csv_file_path: str
        :param: Path of the directory that audio file going to be stored in
        :type path : str
        :param count_down_path: Count Down audio files directory name
        :type count_down_path: str
        :return: True if the download process is successful and False if it is not.
        :rtype: bool
        """

        try:

            # Generate the directory based on the root path
            directory_path = os.path.join(self.ROOT_PATH, path)
            count_down_directory_path = os.path.join(self.ROOT_PATH, count_down_path)

            if not os.path.exists(directory_path):
                print("The directory is not exists.")
                os.mkdir(directory_path)
                print("The directory has been created.")

            audio_file_path = os.path.join(self.ROOT_PATH, csv_file_path)

            if not os.path.isfile(audio_file_path):
                raise FileNotFoundError("The CSV file is not exists. Check the file path")

            # Load the csv file into a pandas DataFrame
            voices_data_frame = pd.read_csv(audio_file_path)

            # Iterate over the rows of the pandas DataFrame to download each audio_countdown
            i = 1
            for _, row in voices_data_frame.iterrows():

                # Read the data record from the repository
                query_builder = f"SELECT *  FROM syn5511444 Where recordId = '{row['recordId']}'"

                # Load the data
                voice_data = self.syn.tableQuery(query_builder)

                voice_data_df = voice_data.asDataFrame()

                dist_path = os.path.join(directory_path, f"{voice_data_df.iloc[0]['audio_audio.m4a']}.m4a")
                count_down_file_path = os.path.join(count_down_directory_path,
                                                    f"{voice_data_df.iloc[0]['audio_countdown.m4a']}.m4a")

                # Check if the count_down version of the audio file exist
                if not os.path.exists(count_down_file_path):
                    # The file has been cleared because of the background noise. Print the message for not downloading
                    # and continue.
                    print(f"The file {voice_data_df.iloc[0]['audio_audio.m4a']} has been cleared because of the "
                          f"background noise.")
                    continue

                if os.path.exists(dist_path):
                    print(f"Count down audio file - {voice_data_df.iloc[0]['audio_countdown.m4a']}.m4a - has already "
                          f"been downloaded. "
                          f"File count is {i}")
                    i += 1
                    continue

                # map the files needed to be downloaded
                voice_map = self.syn.downloadTableColumns(voice_data, ["audio_audio.m4a"])

                for record in voice_map.items():
                    # Generate the target file path
                    dist_path = os.path.join(directory_path, f"{record[0]}.m4a")
                    # Src file path
                    src_path = record[1]
                    # Copy file to the new path
                    copyfile(src_path, dist_path)
                    # Delete the file in the src
                    os.remove(src_path)
                    # Print a comment about downloading the file
                    print(f"Count down audio file - {record[0]}.m4a - downloaded. File count is {i}")
                    i += 1
                    break

            # Process is successful
            return True
        except Exception as ex:
            print(f"A runtime error reported. Error:{str(ex)}")
            # Process failed
            return False
