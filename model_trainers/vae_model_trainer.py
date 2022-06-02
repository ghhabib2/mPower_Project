import os

import numpy as np
import pandas as pd

from ml_models import VAE
from model_trainers import ModelTrainer
import time


class VAETrainer(ModelTrainer):
    def __init__(self, to_read_dir_path
                 , to_store_dir_path
                 , csv_file_name,
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
        """

        # Global variable for holding the training data
        self._training_data = None
        # Global variable for holding the model
        self._model = None
        # Global variable for holding the loss function optimization data

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

        # Iterate over the csv file
        for _, row in csv_data.iterrows():
            # Read the data for each data record

            # Generate the pass to the file
            file_to_load_path = os.path.join(self._TO_READ_PATH,
                                             f"{row['audio_audio']}_{self._segment_number - 1}.npy")

            feature = self._load_file(file_to_read_path=file_to_load_path)

            x_train.append(feature)

        x_train = np.array(x_train)

        self._training_data = x_train[..., np.newaxis]  # Add one dimension to the data

    def _load_file(self, file_to_read_path):
        """
        Read the file to be loaded
        :param (str) file_to_read_path: The path to the file needed to be loaded
        :return: Return the loaded file as numpy array
        """
        return np.load(file_to_read_path)

    def train(self):
        autoencoder = VAE(
            input_shape=(256, 64, 1),
            conv_filters=(512, 256, 128, 64, 32),
            conv_kernels=(3, 3, 3, 3, 3),
            conv_strides=(2, 2, 2, 2, (2, 1)),
            latent_space_dim=128,
            keep_csv_log_dir=f"auto_encoder_model_dir_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        autoencoder.summary()
        autoencoder.compile(self._learning_rate)
        autoencoder.train(self._training_data, self._batch_size, self._epochs)

        return autoencoder

    def save(self):
        self._model.save("model")
