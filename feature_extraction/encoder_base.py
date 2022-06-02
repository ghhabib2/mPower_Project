from feature_extraction import FeatureExtractor
import os
import numpy as np
import pandas as pd
from ml_models import VAE
import itertools

# Root of the system
user_home_path = user_path = os.path.expanduser("~")


class EncoderBaseFeatures(FeatureExtractor):
    def __init__(self,
                 to_read_dir_path,
                 to_store_dir_path,
                 encoder_models_dir_path,
                 encoder_model_weight_file_name,
                 encoder_model_parameters_file_name,
                 csv_file_name,
                 extracted_features_file_name,
                 segment_number,
                 latent_space_dim
                 ):
        """

        Extract the features based on the auto encoder by loading the model

        :param (str) to_read_dir_path: Directory path to read the files
        :param (str) to_store_dir_path: Directory path to store the features
        :param (str) encoder_models_dir_path: Auto encoder model directory path
        :param (str) encoder_model_weight_file_name: Auto encoder model weight file name
        :param (str) encoder_model_parameters_file_name: Auto encoder model parameters file name
        :param (str) csv_file_name: CSV dataset file name
        :param (str) extracted_features_file_name : CSV file address to be saved
        :param (int) segment_number: Segment Number
        :param (int) latent_space_dim: Latent space dimension
        :returns: None
        """

        super().__init__(to_read_dir_path, to_store_dir_path)

        self._ROOT_PATH = os.path.join(user_home_path, "Documents/collected_data_mpower")

        self.encoder_models_dir_path = encoder_models_dir_path
        self.encoder_model_weight_file_name = encoder_model_weight_file_name
        self.encoder_model_parameters_file_name = encoder_model_parameters_file_name
        self.csv_dataset_file_name = csv_file_name
        self.extracted_features_file_name = extracted_features_file_name
        self.segment_number = segment_number
        self.latent_space_dim = latent_space_dim
        self.feature_arrays = None
        self.label_array = None
        self.data_array = None

        # Generate the pass for encoder model
        self._weight_path = \
            os.path.join(
                self.ROOT_PATH,
                f"auto_encoder_models/{self.encoder_models_dir_path}/{self.encoder_model_weight_file_name}")
        self._parameters_path = \
            os.path.join(
                self.ROOT_PATH,
                f"auto_encoder_models/{self.encoder_models_dir_path}/{self.encoder_model_parameters_file_name}")

        # generate the VAE model
        self.vae_model = VAE.load(weights_path=self._weight_path, parameters_path=self._parameters_path)

    def _loader(self, file_path):

        # Generate path to the file
        file_path = os.path.join(self.TO_READ_PATH, file_path)
        # Check if the file exists
        if not os.path.exists(file_path):
            raise IOError("File not found!!")

        # Load the file
        self.feature_vector = np.load(file_path)

    def process(self):

        # Read list of the features into
        csv_file_path = os.path.join(self.TO_READ_PATH, self.csv_dataset_file_name)

        if not os.path.exists(csv_file_path):
            raise IOError(f"{csv_file_path} not found!!")

        csv_data = pd.read_csv(csv_file_path)

        data_array = []
        label_array = []
        # Iterate over the csv file

        print("Please wait, system loading the data ...")
        for _, row in csv_data.iterrows():
            # Read the data for each data record

            # Generate the pass to the file
            file_to_load_path = f"{row['audio_audio']}_{self.segment_number - 1}.npy"
            # Load file
            file_to_be_processed = self._loader(file_to_load_path)
            data_array.append(file_to_be_processed)
            label_array.append(row['medTimepoint'])

        data_array = np.array(data_array)
        self.data_array = data_array[..., np.newaxis]  # Add one dimension to the data
        self.label_array = np.array(label_array)
        # Extract features
        feature_array = []
        print("Please wait, system extracting the features ...")
        for item, label in zip(self.data_array, self.label_array):
            feature_array.append([self.vae_model.encode(data=item), label])

        print("Feature extraction finished.")

        to_be_saved_file_path = os.path.join(self.TO_STORE_PATH, self.extracted_features_file_name)

        self._save_feature(np.array(feature_array), to_be_saved_file_path)

        print("Feature vector saved and ready to be processed.")

    def _save_feature(self, features, file_path):
        # Save the feature file if it is necessary
        features_df = pd.DataFrame(data=features, columns=['data', 'label'])
        features_df.to_csv(file_path)
