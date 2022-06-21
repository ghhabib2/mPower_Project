import datetime
import os
import pickle

import math
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanMetricWrapper
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import keras_tuner as kt
import tensorflow as tf
import time

tf.compat.v1.disable_eager_execution()


# Root of the system
user_home_path = user_path = os.path.expanduser("~")


class VAE:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape,
                 conv_filters_max_size,
                 conv_filters_min_size,
                 conv_kernels_max_size,
                 conv_strides_max_size,
                 latent_space_dim_min,
                 latent_space_dim_max,
                 keep_csv_log_dir,
                 learning_rate=0.0001,
                 reconstruction_loss_weight=1000):

        """

        :param (tuple) input_shape: Input shape of input vectors
        :param (int) conv_filters_max_size: Maximum convolutional layer dimension
        :param (int) conv_filters_min_size: Minimum convolutional layer dimension
        :param (int) conv_strides_max_size: Conv Max Stride Size
        :param (int) latent_space_dim_min: Latent space dimension min size
        :param (int) latent_space_dim_max: Latent space dimension max size
        :param (str) keep_csv_log_dir: Directory for storing the log files
        :param (float) learning_rate: Learning Rate
        :param (int) reconstruction_loss_weight: The reconstruction loss weight
        """

        self._ROOT_PATH = os.path.join(user_home_path, "Documents/collected_data_mpower")

        self.input_shape = input_shape  # [28, 28, 1]
        self.conv_filters_max_size = conv_filters_max_size
        self.conv_filters_min_size = conv_filters_min_size
        self.conv_kernels_max_size = conv_kernels_max_size
        self.conv_strides_max_size = conv_strides_max_size
        self.latent_space_dim_min = latent_space_dim_min  # 2
        self.latent_space_dim_max = latent_space_dim_max  # 2
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.keep_csv_log_dir = os.path.join(self._ROOT_PATH, keep_csv_log_dir)
        self.learning_rate = learning_rate

        # Define the properties to be used later
        self.conv_filters_overall_list = []
        self.encoder = None
        self.decoder = None
        self.model = None
        self.conv_kernels = []
        self.conv_strides = []
        self._shape_before_bottleneck = None
        self._model_input = None

        # Conv Filters
        i = self.conv_filters_max_size
        while i > self.conv_filters_min_size:
            self.conv_filters_overall_list.append(i)
            i = int(i / 2)

        # self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    # def compile(self, learning_rate=0.0001):
    #     optimizer = Adam(learning_rate=learning_rate)
    #     self.model.compile(optimizer=optimizer,
    #                        loss=self._calculate_combined_loss,
    #                        metrics=[MeanMetricWrapper(fn=self._calculate_kl_loss, name="kl_loss"),
    #                                 MeanMetricWrapper(fn=self._calculate_reconstruction_loss,
    #                                                   name="reconstruction_loss")])

    def train(self, x_train, batch_size, num_epochs):

        model_check_point_dir_path = os.path.join(self.keep_csv_log_dir, "models")

        # Create the folder for storing the check_point_models if it is not exist
        if not os.path.isdir(model_check_point_dir_path):
            os.makedirs(model_check_point_dir_path)

        # Model Tuner
        tuner = kt.RandomSearch(
            self._build,
            objective="val_loss",
            max_trials=100,
            seed=0,
            hyperparameters=None,
            tune_new_entries=True,
            allow_new_entries=True,
            overwrite=True,
            directory=model_check_point_dir_path,
            project_name='encoder_tuning_proj')

        # Show a summary
        tuner.search_space_summary()

        # EarlyStopping
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=5,
            verbose=1,
            mode='min',
            baseline=None,
            restore_best_weights=False
        )

        # call_backs = [early_stopping_callback]

        # Tune the hps
        tuner.search(x_train,
                     x_train,
                     epochs=100,
                     validation_split=0.2, callbacks=[early_stopping_callback])
        # Show the result summary
        tuner.results_summary()

        # Check if the directory exists

        # if not os.path.isdir(self.keep_csv_log_dir):
        #     os.makedirs(self.keep_csv_log_dir)
        # # CSVLogger
        # # Add the necessary information for storing the training log
        # csv_logger_callback = CSVLogger(
        #     os.path.join(self.keep_csv_log_dir,
        #                  f"log_{time.strftime('%Y%m%d-%H%M%S')}_dim{self.latent_space_dim}.csv"))
        # call_backs.append(csv_logger_callback)

        # # ModelCheckpoint
        # model_check_point_dir_path = os.path.join(self.keep_csv_log_dir, "models")
        #
        # # Create the folder for storing the check_point_models if it is not exist
        # if not os.path.isdir(model_check_point_dir_path):
        #     os.makedirs(model_check_point_dir_path)

        # # File path and template for checkpoint models
        # model_check_point_file_path = os.path.join(model_check_point_dir_path,
        #                                            "{epoch:02d}-{loss:.2f}.hdf5")
        # model_checkpoit_callback = ModelCheckpoint(
        #     filepath=model_check_point_file_path,
        #     save_weights_only=True,
        #     monitor='loss',
        #     mode='min',
        #     save_best_only=True
        # )
        # call_backs.append(model_checkpoit_callback)

        # # EarlyStopping
        # early_stopping_callback = EarlyStopping(
        #     monitor='loss',
        #     min_delta=0,
        #     patience=5,
        #     verbose=1,
        #     mode='min',
        #     baseline=None,
        #     restore_best_weights=False
        # )
        #
        # call_backs.append(early_stopping_callback)
        #
        # # Tensorboard
        # model_tensorboard_dir_callback = \
        #     os.path.join(self.keep_csv_log_dir,
        #                  f"tensor_board_log_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        #
        # # Check the directory and make it if it is not exist.
        # if not os.path.isdir(model_tensorboard_dir_callback):
        #     os.makedirs(model_tensorboard_dir_callback)
        #
        # tensorboard_callback = TensorBoard(log_dir=model_tensorboard_dir_callback, histogram_freq=1)
        #
        # call_backs.append(tensorboard_callback)
        #
        # self.model.fit(x_train,
        #                x_train,
        #                batch_size=batch_size,
        #                callbacks=call_backs,
        #                epochs=num_epochs,
        #                shuffle=True)

    def save(self):
        # generate the model directory name
        model_dir_name = f"model_auto_encoder_{time.strftime('%Y%m%d-%H%M%S')}_dim_{self.latent_space_dim}"
        model_dir_path = os.path.join(self._ROOT_PATH, f"auto_encoder_models/{model_dir_name}")
        self._create_folder_if_it_doesnt_exist(model_dir_path)
        self._save_parameters(model_dir_path)
        self._save_weights(model_dir_path)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def encode(self, data):
        """
        Encode the data and generate the
        :param data: Data to be encoded
        :return: Encoded data
        """
        return self.encoder.predict(data)

    def decode(self, data):
        """
        Decode the data

        :param data: Data to be decoded
        :return: Reconstructed data
        """
        return self.decoder.predict(data)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, weights_path, parameters_path):

        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)

        autoencoder.load_weights(weights_path)
        return autoencoder

    def _calculate_combined_loss(self, y_target, y_predicted):
        reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
        kl_loss = self._calculate_kl_loss(y_target, y_predicted)
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss \
                        + kl_loss
        return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted

        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_mu(self, y_target, y_predicted):
        return self.mu

    def _calculate_log_variance(self, y_target, y_predicted):
        return K.exp(self.log_variance / 2)

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)
        return kl_loss

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):

        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim,
            "",
            self.reconstruction_loss_weight
        ]
        save_path = os.path.join(save_folder, f"parameters_{time.strftime('%Y%m%d-%H%M%S')}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, f"weights_{time.strftime('%Y%m%d-%H%M%S')}.hdf5")
        self.model.save_weights(save_path)

    def _build(self, hp):

        # Generate the list of conv layers for process
        temp_conv_filters = hp.Choice('best_conv_filter',
                                      values=self.conv_filters_overall_list)
        temp_kernel_size = hp.Int('best_kernel_size',
                                  min_value=1,
                                  max_value=self.conv_kernels_max_size,
                                  step=2)

        temp_conv_stride_size = self.conv_strides_max_size

        temp_latent_space_dim = hp.Int('best_latent_space_dim',
                                       min_value=self.latent_space_dim_min,
                                       max_value=self.latent_space_dim_max)

        # Set the parameters for hyperparameter optimization
        self.latent_space_dim = temp_latent_space_dim
        self.conv_filters = []
        i = temp_conv_filters
        while i >= self.conv_filters_min_size:
            self.conv_filters.append(i)
            i = int(i / 2)

        self.conv_filters = tuple(self.conv_filters)
        self._num_conv_layers = len(self.conv_filters)

        self.conv_kernels = []
        self.conv_strides = []
        for i in range(self._num_conv_layers):
            self.conv_kernels.append(temp_kernel_size)
            if i == self._num_conv_layers - 1:
                self.conv_strides.append((temp_conv_stride_size, 1))
            else:
                self.conv_strides.append(temp_conv_stride_size)

        # Convert List to the Tuple
        self.conv_kernels = tuple(self.conv_kernels)

        # Kernel Size
        self.conv_strides = tuple(self.conv_strides)

        self._build_encoder()
        self._build_decoder()

        temp_model = self._build_autoencoder()

        temp_model.summary()

        optimizer = Adam(learning_rate=self.learning_rate)
        temp_model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss,
                           metrics=[MeanMetricWrapper(fn=self._calculate_kl_loss, name="kl_loss"),
                                    MeanMetricWrapper(fn=self._calculate_reconstruction_loss,
                                                      name="reconstruction_loss"),
                                    MeanMetricWrapper(fn=self._calculate_mu, name="mu"),
                                    MeanMetricWrapper(fn=self._calculate_log_variance, name="std")])

        return temp_model

    def _build_autoencoder(self):
        """
        Return the model to be trained

        :return: Return the model to be trained
        """
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        return Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")
        return self.decoder

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        # num_neurons = np.prod(self._shape_before_bottleneck)
        num_neurons = math.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the
        # first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):

        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Create all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Add a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck with Guassian sampling (Dense
        layer).
        """

        # Latent Space Dimension Hypermarket

        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim,
                                  name="log_variance")(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args

            epsilon = K.random_normal(shape=K.shape(self.mu),
                                      mean=0,
                                      stddev=1.)

            sampled_point = mu + K.exp(log_variance / 2) * epsilon

            return sampled_point

        x = Lambda(sample_point_from_normal_distribution,
                   name="encoder_output")([self.mu, self.log_variance])
        return x
