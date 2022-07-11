import os

import numpy as np
import pandas as pd

from ml_models import VAEPURE
from model_trainers import ModelTrainer
import time


class VAEPURETrainer(ModelTrainer):
    def __init__(self,
                 to_read_dir_path,
                 to_store_dir_path,
                 csv_file_name,
                 latent_space_dim=128,
                 learning_rate=0.0005,
                 batch_size=64,
                 epochs=200,
                 segment_number=2):
        """
        Read the configuration of the network before starting the learning process.


        :param (str) csv_file_name: The CSV file name to be loaded
        :param (str) to_read_dir_path: The directory path for reading the data
        :param (str) to_store_dir_path: The directory path for saving any sort of data.
        :param (int) segment_number: The target segment number for training.
        :param (int) latent_space_dim: Latent space dimension.
        """

        # Global variable for holding the training data
        self._training_data = None
        # Global variable for holding the model
        self._model = None
        # Global variable for holding the loss function optimization data
        self._latent_space_dim = latent_space_dim
        self._csv_file_name = csv_file_name
        self._segment_number = segment_number
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._epochs = epochs

        super().__init__(to_read_dir_path=to_read_dir_path,
                         to_store_dir_path=to_store_dir_path,
                         learning_rate=learning_rate,
                         batch_size=batch_size,
                         epochs=epochs)

    def load(self):
        """
        Loading the data for training

        :return: Nothing
        :rtype: None
        """
        # Generate file path
        csv_file_path = os.path.join(self._TO_READ_PATH, self._csv_file_name)

        # Check if the csv file loaded properly
        if not os.path.exists(csv_file_path):
            raise IOError("CSV file not found!!")

        # Load the file content into a pandas dataframe
        csv_data = pd.read_csv(csv_file_path)

        x_train = []

        print("Please wait while system loading the data ...")
        # Iterate over the csv file
        minimum_values = None
        maximum_values = None
        for _, row in csv_data.iterrows():
            # Read the data for each data record

            # Generate the pass to the file
            file_to_load_path = os.path.join(self._TO_READ_PATH,
                                             f"{row['audio_audio']}/{row['audio_audio']}_{self._segment_number}.npy")
            try:
                feature = self._load_file(file_to_read_path=file_to_load_path)
                if minimum_values is None:
                    minimum_values, _ = feature.shape
                    maximum_values, _ = feature.shape
                else:
                    minimum_values = min(minimum_values, feature.shape[0])
                    maximum_values = max(maximum_values, feature.shape[0])
                x_train.append(feature)
            except Exception as ex:
                print(f"The feature not loaded with the following exception \n\n{str(ex)}")
                continue

        # Find the mimumul value
        # TODO This changes are made because of the input file we have for the network with (433, 16) shape. The target
        # Shape is (448, 16)
        # for index, item in enumerate(x_train):
        #     # Cut the useless part of the array
        #     # temp_array = np.concatenate([item[:433,:], temp_zero_array], axis=0)
        #     x_train[index] = item

        x_train = np.array(x_train)

        self._training_data = x_train[..., np.newaxis]  # Add one dimension to the data
        print("Data has been loaded successfully.")

    def _load_file(self, file_to_read_path):
        """
        Read the file to be loaded
        :param (str) file_to_read_path: The path to the file needed to be loaded
        :return: Return the loaded file as numpy array
        """
        return np.load(file_to_read_path)

    def train(self):

        autoencoder = VAEPURE(
            input_shape=(256, 32, 1),
            latent_space_dim_max=3,
            latent_space_dim_min=2,
            conv_filters_max_size=64,
            conv_filters_min_size=16,
            conv_kernels_max_size=7,
            conv_strides_max_size=2,
            keep_csv_log_dir=f"trainings/auto_encoder_model_dir_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        # autoencoder.summary()
        # autoencoder.compile(self._learning_rate)
        autoencoder.build()
        autoencoder.train(self._training_data, self._batch_size, self._epochs)
        autoencoder.save()

        return autoencoder

    def save(self):
        self._model.save("model")
