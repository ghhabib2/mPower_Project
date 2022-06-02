import os
import abc

# Root of the system
user_home_path = user_path = os.path.expanduser("~")

class ModelTrainer(object):
    """
    Abstract class that represent the interface of a the Model trainer class
    """

    def __init__(self, to_read_dir_path, to_store_dir_path, learning_rate=0.0005, batch_size= 64, epochs=150):
        """

        :param (str) to_read_dir_path: The directory path for reading the data
        :param (str) to_store_dir_path: The directory path for saving any sort of data.
        :param (float) learning_rate: Learning rate for the trainer (Alpha value)
        :param (int) batch_size: Batch size for the trainer
        :param (int): Number of epoches for a model to be trained.
        """
        self._ROOT_PATH = os.path.join(user_home_path, "Documents/collected_data_mpower")
        self._TO_READ_PATH = os.path.join(self._ROOT_PATH, to_read_dir_path)
        self._TO_STORE_PATH = os.path.join(self._ROOT_PATH, to_store_dir_path)
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._epochs = epochs
        self._model = None

    @abc.abstractmethod
    def load(self):
        """
        Load the data to be used for training and validation/testing

        :return: Return the extracted data
        """

        raise NotImplemented("This method has not implemented yet.")

    @abc.abstractmethod
    def train(self):
        """
        Performing the training

        :return: Return the trained model
        """

        raise NotImplemented("This method has not implemented yet.")

    @abc.abstractmethod
    def save(self):
        """
        Store the model if it is necessary to do so
        :return:
        """
        raise NotImplemented("This method has not implemented yet.")

    @abc.abstractmethod
    def model_validation(self):
        """
        Performing the model validation

        :return: Return either nothing or the validation result
        """

        raise NotImplemented("This method has not implemented yet.")

