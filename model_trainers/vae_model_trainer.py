import os

import numpy as np
import pandas as pd

from ml_models import VAE
from model_trainers import ModelTrainer
import time


class VAETrainer(ModelTrainer):
    def __init__(self,
                 to_read_dir_path,
                 to_store_dir_path,
                 csv_file_name,
                 latent_space_dim=128,
                 learning_rate=0.0005,
                 batch_size=64,
                 epochs=150,
                 segment_number=1):
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

        for _, row in csv_data.iterrows():
            # Read the data for each data record

            # Generate the pass to the file
            file_to_load_path = os.path.join(self._TO_READ_PATH,
                                             f"{row['audio_audio']}/{row['audio_audio']}_{self._segment_number}.npy")
            try:
                feature = self._load_file(file_to_read_path=file_to_load_path)
                x_train.append(feature)
            except Exception as ex:
                print(f"The feature not loaded with the following exception \n\n{str(ex)}")
                continue

        # Find the mimumul value
        for index, item in enumerate(x_train):
            # Cut the useless part of the array
            x_train[index] = item[:, :192]

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
        autoencoder = VAE(
            input_shape=(256, 192, 1),
            latent_space_dim_max=3,
            latent_space_dim_min=2,
            conv_filters_max_size=1024,
            conv_filters_min_size=32,
            conv_kernels_max_size=5,
            conv_strides_max_size=2,
            keep_csv_log_dir=f"auto_encoder_model_dir_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        # autoencoder.summary()
        # autoencoder.compile(self._learning_rate)
        autoencoder.train(self._training_data, self._batch_size, self._epochs)

        return autoencoder

    def save(self):
        self._model.save("model")
