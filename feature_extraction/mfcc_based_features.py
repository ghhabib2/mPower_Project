"""
1- load a file
2- pad the signal (if necessary)
3- extracting log spectrogram from signal
4- normalise spectrogram
5- save the normalised spectrogram

PreprocessingPipeline
"""
import gc
import os
import pickle
import math
import pandas as pd

from feature_extraction import FeatureExtractor
import librosa
import numpy as np


class Padder:
    """Padder is responsible to apply padding to an array."""

    def __init__(self, mode="constant"):
        self.mode = mode

    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (num_missing_items, 0),
                              mode=self.mode)
        return padded_array

    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,
                              (0, num_missing_items),
                              mode=self.mode)
        return padded_array


class MFCCFeatureExtractor:
    """
    LogSpectrogramExtractor extracts log spectrogram's (in dB) from a
    time-series signal.
    """

    def __init__(self, hop_length,
                 n_mfcc,
                 sr,
                 expected_num_mfcc_vectors_per_segment, is_norm=True):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.is_norm = is_norm
        self.expected_num_mfcc_vectors_per_segment = expected_num_mfcc_vectors_per_segment

    def extract(self, signal):

        s = librosa.feature.melspectrogram(y=signal, sr=self.sr, n_mels=128, fmax=8000)

        if self.is_norm:
            # mfcc = librosa.feature.mfcc(S=librosa.power_to_db(s), n_mfcc=self.n_mfcc, norm='ortho')
            mfcc = librosa.feature.mfcc(y=signal,
                                        sr=self.sr,
                                        n_fft=2048,
                                        hop_length=512,
                                        n_mfcc=self.n_mfcc,
                                        norm='ortho')

        else:
            mfcc = librosa.feature.mfcc(y=signal,
                                        sr=self.sr,
                                        n_fft=2048,
                                        hop_length=512,
                                        n_mfcc=self.n_mfcc)

        if len(mfcc.T) == self.expected_num_mfcc_vectors_per_segment:
            return mfcc
        else:
            raise RuntimeError("Not correct size")


class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normalisation to an array."""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class MFCCExtractor(FeatureExtractor):
    """PreprocessingPipeline processes audio files in a directory, applying
    the following steps to each file:
        1- load a file
        2- pad the signal (if necessary)
        3- extracting log spectrogram from signal
        4- normalise spectrogram
        5- save the normalised spectrogram

    Storing the min max values for all the log spectrograms.
    """

    def __init__(self, to_read_dir_path,
                 to_store_dir_path,
                 dataset_csv_file,
                 sample_rate=22050,
                 hop_length=512,
                 segment_duration=1,
                 n_mfcc=13,
                 is_norm=True,
                 mono=True):
        """ Extract Spectrogram Features

        :param (int) sample_rate: Sample Rate
        :param (float) segment_duration: Duration of each feature sample
        :param (bool) mono: Flag check if the audio file should be loaded in Mono mode or in Stereo Mode.
        :param (int) n_mfcc : Number of MFCC coefficients
        :param (int) hope_length : Hope length
        """

        super().__init__(to_read_dir_path, to_store_dir_path)

        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._num_expected_samples = None
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.mono = mono
        self.dataset_csv_file = dataset_csv_file
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.is_norm = is_norm

    # Define the loader
    def _loader(self, file_path):
        signal, _ = librosa.load(file_path,
                                 sr=self.sample_rate,
                                 mono=self.mono)
        return signal

    def process(self):

        # Check if the csv file exists
        if len(self.dataset_csv_file) > 0:
            csv_file_path = os.path.join(self.TO_READ_PATH, self.dataset_csv_file)
        else:
            raise IOError("You need to path the `dataset_csv_file` value.")

        # Read the file content into a pandas dataframe
        voices_dataset = pd.read_csv(csv_file_path)

        # Iterate over the data in the pandas dataframe

        i = 1
        row_collection = []
        for _, row in voices_dataset.iterrows():
            print(f"{i}- Extracting features for {row['audio_audio.m4a']}")

            # Generate the path for the file
            audio_file_path = os.path.join(self.TO_READ_PATH, f"{row['audio_audio.m4a']}.m4a")

            # Check if the file exists in the folder
            if not os.path.exists(audio_file_path):
                print("The file deleted due to noise presentation!!. Feature extraction failed")
                continue

            # Process file and generate the feature vectors
            try:
                self._process_file(file_name=row["audio_audio.m4a"],
                                   file_path=audio_file_path,
                                   store_path=self.TO_STORE_PATH)
            except Exception as ex:
                print(f"The process of feature extraction failed for this file for the following reason:\n\n{str(ex)}")
                continue

            print(f"Features for {row['audio_audio.m4a']} has been extracted.")

            # Add the row to the list of the row collections
            row_collection.append([row['healthCode'],
                                   row['audio_audio.m4a'],
                                   row["medTimepoint"],
                                   row['createdOn']])
            i += 1

            gc.collect()

        # Generate the new file with the extracted features rows
        # Add the rows to feature collection file
        # creat a temp file for storing the features basic information.
        temp_dataset_df = pd.DataFrame(data=np.array(row_collection), columns=['healthCode',
                                                                               'audio_audio',
                                                                               'medTimepoint',
                                                                               'createdOn'])

        extracted_features_file_path = os.path.join(self.TO_STORE_PATH, "extracted_features_csv.csv")

        print("Storing the data into csv dataset")

        temp_dataset_df.to_csv(extracted_features_file_path, index=False)

        print("Feature extraction process has been finished.")

    def _process_file(self, file_name, file_path, store_path):
        """
        Process the file and extract the features

        :param (str) file_name: It is kind of obvious what it is
        :param (str) file_path: File path to be processed
        :param (str) store_path: Path of the feature file to be stored
        :return: Nothing
        :rtype: None
        """
        # Load the target signal
        signal = self._loader(file_path)

        # Calculate the file  duration
        signal_duration = librosa.get_duration(y=signal, sr=self.sample_rate)

        # Calculate the sample per track
        # Calculate the number of samples per segment
        # Calculate the sample per track value
        sample_per_track = signal_duration * self.sample_rate
        # Calculate the number of sample segments

        number_of_samples_per_segment = 3 * self.sample_rate
        # Calculate the number of possible segments for the file
        num_segments = int(sample_per_track // number_of_samples_per_segment)
        # Calculate the expected mfcc vector length
        expected_num_mfcc_vectors_per_segment = math.ceil(number_of_samples_per_segment / self.hop_length)

        # Check if any folder with the name of the file exists in the store folder
        file_directory_path = os.path.join(store_path, str(file_name))

        if not os.path.isdir(file_directory_path):
            # Create directory
            os.makedirs(file_directory_path)

        i = 1

        # Set the segment number to 2
        start_sample = int(self.sample_rate * 2)
        finish_sample = int(start_sample + number_of_samples_per_segment)
        sample_per_track = signal[start_sample:finish_sample]
        feature = MFCCFeatureExtractor(hop_length=self.hop_length,
                                       n_mfcc=self.n_mfcc,
                                       sr=self.sample_rate,
                                       is_norm=self.is_norm,
                                       expected_num_mfcc_vectors_per_segment=expected_num_mfcc_vectors_per_segment)\
            .extract(sample_per_track)

        # If the normalization selected the mfcc built-in normalization mechanism should apply to the mfcces
        # If not MinMaxNormalization could apply
        if self.is_norm:
            norm_feature = feature
        else:
            norm_feature = MinMaxNormaliser(0, 1).normalise(feature)

        # Generate file path
        save_file_path = os.path.join(file_directory_path, f"{file_name}_2.npy")
        # Save the feature vector
        self._save_feature(norm_feature, save_file_path)
        # Generate the file path for storing max and min values
        save_file_path = os.path.join(file_directory_path, f"{file_name}_2.pkl")
        self._save_min_max_values(save_file_path, feature.min(), feature.max())

        # for s in range(num_segments):
        #
        #     # Separate the segment
        #     start_sample = int(number_of_samples_per_segment * s)
        #     finish_sample = int(start_sample + number_of_samples_per_segment)
        #     sample_per_track = signal[start_sample:finish_sample]
        #
        #     # # Check if the padding is necessary to be added
        #     # if self._is_padding_necessary(sample_per_track, num_expected_samples=num_expected_samples):
        #     #     # Add the padding to the signal
        #     #     signal = self._apply_padding(sample_per_track)
        #
        #     # Extract the features
        #     feature = MFCCFeatureExtractor(hop_length=self.hop_length,
        #                                    n_mfcc=self.n_mfcc,
        #                                    sr=self.sample_rate,
        #                                    is_norm=self.is_norm,
        #                                    expected_num_mfcc_vectors_per_segment=expected_num_mfcc_vectors_per_segment)\
        #         .extract(sample_per_track)
        #
        #     # If the normalization selected the mfcc built-in normalization mechanism should apply to the mfcces
        #     # If not MinMaxNormalization could apply
        #     if self.is_norm:
        #         norm_feature = feature
        #     else:
        #         norm_feature = MinMaxNormaliser(0, 1).normalise(feature)
        #
        #     norm_feature = np.array(norm_feature).T
        #
        #     # Generate file path
        #     save_file_path = os.path.join(file_directory_path, f"{file_name}_{i}.npy")
        #     # Save the feature vector
        #     self._save_feature(norm_feature, save_file_path)
        #     # Generate the file path for storing max and min values
        #     save_file_path = os.path.join(file_directory_path, f"{file_name}_{i}.pkl")
        #     self._save_min_max_values(save_file_path, feature.min(), feature.max())
        #
        #     i += 1

    def _save_feature(self, feature, file_path):
        """
        Save the feature based on the file path

        :param (np.ndarray) feature: Feature to be saved
        :param (str) file_path: Feature file path as string
        :return: Nothing
        :rtype: None
        """
        np.save(file_path, feature)

    def _save_min_max_values(self, file_path, min_val, max_val):
        """
        Save the min-max value

        :param file_path: File path to store the min-max values
        :param (float) min_val: Min value
        :param (float) max_val: Max value
        :return:
        """
        value_to_be_saved = {
            "min": min_val,
            "max": max_val
        }

        self._save(value_to_be_saved, file_path)

    @staticmethod
    def _save(data, save_path):
        """
        Min-Max value saver

        :param data: Data to be saved
        :param save_path: Path of the file
        :return:
        """
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _is_padding_necessary(self, signal, num_expected_samples):
        """
        Identify if the padding necessary for the signal based on the expected number of samples.

        :param (np.ndarray) signal: Target Signal
        :param (float) num_expected_samples : Number of expected samples
        :return: True if it is necessary to add padding and false if it is not.
        :rtype: bool
        """
        if len(signal) < num_expected_samples:
            return True
        return False

    def _apply_padding(self, signal):
        """
        Apply padding to the target signal
        :param (np.ndarray) signal: Target signal
        :return: Return the target signal after adding the padding.
        :rtype: np.ndarray
        """
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

# if __name__ == "__main__":
#
#
#     SPECTROGRAMS_SAVE_DIR = "/home/valerio/datasets/fsdd/spectrograms/"
#     MIN_MAX_VALUES_SAVE_DIR = "/home/valerio/datasets/fsdd/"
#     FILES_DIR = "/home/valerio/datasets/fsdd/audio/"
#
#     # instantiate all objects
#     #loader = Loader(SAMPLE_RATE, DURATION, MONO)
#     padder = Padder()
#     log_spectrogram_extractor =
#     min_max_normaliser = MinMaxNormaliser(0, 1)
#     saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
#
#     preprocessing_pipeline = Spectrogram_Extractor()
#     preprocessing_pipeline.loader = loader
#     preprocessing_pipeline.padder = padder
#     preprocessing_pipeline.extractor = log_spectrogram_extractor
#     preprocessing_pipeline.normaliser = min_max_normaliser
#     preprocessing_pipeline.saver = saver
#
#     preprocessing_pipeline.process(FILES_DIR)
