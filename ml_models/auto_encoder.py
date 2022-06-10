import datetime
import os
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanMetricWrapper
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
import numpy as np
import tensorflow as tf
import time

tf.compat.v1.disable_eager_execution()

# Root of the system
user_home_path = user_path = os.path.expanduser("~")


# def _calculate_kl_loss(self, y_target, y_predicted):
#     kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
#                            K.exp(self.log_variance), axis=1)


class VAE:
    """
    VAE represents a Deep Convolutional variational autoencoder architecture
    with mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim,
                 keep_csv_log_dir,
                 reconstruction_loss_weight=1000):

        """

        :param (tuple) input_shape: Input shape of input vectors
        :param (tuple) conv_filters: Convolutional layers number and dimension
        :param (tuple) conv_kernels: Kernel of the convolution layers
        :param (tuple) conv_strides: stride of each convolution layer
        :param (int) latent_space_dim: Latent space dimension
        :param (str) keep_csv_log_dir: Directory for storing the log files
        :param (int) reconstruction_loss_weight: The reconstruction loss weight
        """

        self._ROOT_PATH = os.path.join(user_home_path, "Documents/collected_data_mpower")

        self.input_shape = input_shape  # [28, 28, 1]
        self.conv_filters = conv_filters  # [2, 4, 8]
        self.conv_kernels = conv_kernels  # [3, 5, 3]
        self.conv_strides = conv_strides  # [1, 2, 2]
        self.latent_space_dim = latent_space_dim  # 2
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.keep_csv_log_dir = os.path.join(self._ROOT_PATH, keep_csv_log_dir)
        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self._calculate_combined_loss,
                           metrics=[MeanMetricWrapper(fn=self._calculate_kl_loss, name="kl_loss"),
                                    MeanMetricWrapper(fn=self._calculate_reconstruction_loss,
                                                      name="reconstruction_loss")])

    def train(self, x_train, batch_size, num_epochs):
        call_backs = []

        # Check if the directory exists

        if not os.path.isdir(self.keep_csv_log_dir):
            os.makedirs(self.keep_csv_log_dir)
        # CSVLogger
        # Add the necessary information for storing the training log
        csv_logger_callback = CSVLogger(
            os.path.join(self.keep_csv_log_dir,
                         f"log_{time.strftime('%Y%m%d-%H%M%S')}_dim{self.latent_space_dim}.csv"))
        call_backs.append(csv_logger_callback)

        # ModelCheckpoint
        model_check_point_dir_path = os.path.join(self.keep_csv_log_dir, "models")

        # Create the folder for storing the check_point_models if it is not exist
        if not os.path.isdir(model_check_point_dir_path):
            os.makedirs(model_check_point_dir_path)

        # File path and template for checkpoint models
        model_check_point_file_path = os.path.join(model_check_point_dir_path,
                                                   "{epoch:02d}-{val_loss:.2f}.hdf5")
        model_checkpoit_callback = ModelCheckpoint(
            filepath=model_check_point_file_path,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True
        )
        call_backs.append(model_checkpoit_callback)

        # EarlyStopping
        early_stopping_callback = EarlyStopping(
            monitor='loss',
            min_delta=0,
            patience=5,
            verbose=1,
            mode='min',
            baseline=None,
            restore_best_weights=False
        )

        call_backs.append(early_stopping_callback)

        # Tensorboard
        model_tensorboard_dir_callback = \
            os.path.join(self.keep_csv_log_dir,
                         f"tensor_board_log_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")

        # Check the directory and make it if it is not exist.
        if not os.path.isdir(model_tensorboard_dir_callback):
            os.makedirs(model_tensorboard_dir_callback)

        tensorboard_callback = TensorBoard(log_dir=model_tensorboard_dir_callback, histogram_freq=1)

        call_backs.append(tensorboard_callback)

        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       callbacks=call_backs,
                       epochs=num_epochs,
                       shuffle=True)

        

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

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
                               K.exp(self.log_variance), axis=1)
        return kl_loss

    def _calculate_kl_loss_metric(self, FACTOR=1):
        def _calculate_kl_loss(y_target, y_predicted):
            kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
                                   K.exp(self.log_variance), axis=1) * FACTOR
            return kl_loss

        return _calculate_kl_loss

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

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)  # [1, 2, 4] -> 8
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
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim,
                                  name="log_variance")(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.,
                                      stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(sample_point_from_normal_distribution,
                   name="encoder_output")([self.mu, self.log_variance])
        return x

# -----
# import os
# import pickle
#
# # from tensorflow.keras import Model
# # from tensorflow.keras.layers import InputLayer, Input, Conv2D, ReLU, BatchNormalization, \
# #     Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
# # from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
# #
# # from tensorflow.keras.optimizers import Adam
# # from tensorflow.keras.losses import MeanSquaredError
#
# import numpy as np
# import tensorflow
# import tensorflow as tf
# from tensorflow.keras.layers import InputLayer, Conv2D, ReLU, BatchNormalization, Dense, Flatten, Reshape, \
#     Conv2DTranspose, Activation
# from tensorflow.keras import backend as K
# from tensorflow.keras import Sequential
# import time
#
# tf.compat.v1.disable_eager_execution()
#
# # Root of the system
# user_home_path = user_path = os.path.expanduser("~")
#
#
# class VAE:
#     """
#     VAE represents a Deep Convolutional variational autoencoder architecture
#     with mirrored encoder and decoder components.
#     """
#
#     def __init__(self,
#                  input_shape,
#                  conv_filters,
#                  conv_kernels,
#                  conv_strides,
#                  latent_space_dim,
#                  keep_csv_log_dir,
#                  reconstruction_loss_weight=1000):
#
#         """
#
#         :param (tuple) input_shape: Input shape of input vectors
#         :param (tuple) conv_filters: Convolutional layers number and dimension
#         :param (tuple) conv_kernels: Kernel of the convolution layers
#         :param (tuple) conv_strides: stride of each convolution layer
#         :param (int) latent_space_dim: Latent space dimension
#         :param (str) keep_csv_log_dir: Directory for storing the log files
#         :param (int) reconstruction_loss_weight: The reconstruction loss weight
#         """
#
#         self._ROOT_PATH = os.path.join(user_home_path, "Documents/collected_data_mpower")
#
#         self.input_shape = input_shape  # [28, 28, 1]
#         self.conv_filters = conv_filters  # [2, 4, 8]
#         self.conv_kernels = conv_kernels  # [3, 5, 3]
#         self.conv_strides = conv_strides  # [1, 2, 2]
#         self.latent_space_dim = latent_space_dim  # 2
#         self.reconstruction_loss_weight = reconstruction_loss_weight
#         self.keep_csv_log_dir = os.path.join(self._ROOT_PATH, keep_csv_log_dir)
#         self.encoder: Sequential = Sequential()
#         self.decoder: Sequential = Sequential()
#         self.model: Sequential = Sequential()
#
#         self._num_conv_layers = len(conv_filters)
#         self._shape_before_bottleneck = None
#         self._model_input = None
#
#         self._build()
#
#     def summary(self):
#         self.encoder.summary()
#         self.decoder.summary()
#         self.model.summary()
#
#     def compile(self, learning_rate=0.0001):
#         optimizer = Adam(learning_rate=learning_rate)
#         self.model.compile(optimizer=optimizer,
#                            loss=self._calculate_combined_loss,
#                            metrics=[self._calculate_reconstruction_loss, self._calculate_kl_loss])
#
#     def train(self, x_train, batch_size, num_epochs):
#         call_backs = []
#         if self.latent_space_dim:
#             # Check if the directory exists
#
#             if not os.path.isdir(self.keep_csv_log_dir):
#                 os.mkdir(self.keep_csv_log_dir)
#             # Add the necessary information for storing the training log
#             csv_logger_callback = CSVLogger(
#                 os.path.join(self.keep_csv_log_dir,
#                              f"log_{time.strftime('%Y%m%d-%H%M%S')}_dim{self.latent_space_dim}.csv"))
#             call_backs.append(csv_logger_callback)
#         self.model.fit(x_train,
#                        x_train,
#                        batch_size=batch_size,
#                        callbacks=call_backs,
#                        epochs=num_epochs,
#                        shuffle=True)
#
#     def save(self):
#         # generate the model directory name
#         model_dir_name = f"model_auto_encoder_{time.strftime('%Y%m%d-%H%M%S')}_dim_{self.latent_space_dim}"
#         model_dir_path = os.path.join(self._ROOT_PATH, f"auto_encoder_models/{model_dir_name}")
#         self._create_folder_if_it_doesnt_exist(model_dir_path)
#         self._save_parameters(model_dir_path)
#         self._save_weights(model_dir_path)
#
#     def load_weights(self, weights_path):
#         self.model.load_weights(weights_path)
#
#     def reconstruct(self, images):
#         latent_representations = self.encoder.predict(images)
#         reconstructed_images = self.decoder.predict(latent_representations)
#         return reconstructed_images, latent_representations
#
#     @classmethod
#     def load(cls, weights_path, parameters_path):
#
#         with open(parameters_path, "rb") as f:
#             parameters = pickle.load(f)
#         autoencoder = VAE(*parameters)
#
#         autoencoder.load_weights(weights_path)
#         return autoencoder
#
#     def _calculate_combined_loss(self, y_target, y_predicted):
#         reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
#         kl_loss = self._calculate_kl_loss
#         combined_loss = self.reconstruction_loss_weight * reconstruction_loss \
#                         + kl_loss
#         return combined_loss
#
#     # def _calculate_combined_loss(self, y_target, y_predicted):
#     #     reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
#     #     kl_loss = self._calculate_kl_loss(y_target, y_predicted)
#     #     combined_loss = self.reconstruction_loss_weight * reconstruction_loss \
#     #                     + kl_loss
#     #     return combined_loss
#
#     def _calculate_reconstruction_loss(self, y_target, y_predicted):
#         error = y_target - y_predicted
#         reconstruction_loss = tf.reduce_mean(tf.square(error), axis=[1, 2, 3])
#         return reconstruction_loss
#
#     def _calculate_reconstruction_loss(self, y_target, y_predicted):
#         error = y_target - y_predicted
#         reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
#         return reconstruction_loss
#
#     def _calculate_kl_loss(self):
#         kl_loss = -0.5 * tf.reduce_sum(1 + self.log_variance - tf.square(self.mu) - tf.exp(self.log_variance),
#                                        axis=1)
#         return kl_loss
#
#     # def _calculate_kl_loss(self, y_target, y_predicted):
#     #     kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
#     #                            K.exp(self.log_variance), axis=1)
#     #     return kl_loss
#
#     def _create_folder_if_it_doesnt_exist(self, folder):
#         if not os.path.exists(folder):
#             os.makedirs(folder)
#
#     def _save_parameters(self, save_folder):
#
#         parameters = [
#             self.input_shape,
#             self.conv_filters,
#             self.conv_kernels,
#             self.conv_strides,
#             self.latent_space_dim,
#             "",
#             self.reconstruction_loss_weight
#         ]
#         save_path = os.path.join(save_folder, f"parameters_{time.strftime('%Y%m%d-%H%M%S')}.pkl")
#         with open(save_path, "wb") as f:
#             pickle.dump(parameters, f)
#
#     def _save_weights(self, save_folder):
#         save_path = os.path.join(save_folder, f"weights_{time.strftime('%Y%m%d-%H%M%S')}.h5")
#         self.model.save_weights(save_path)
#
#     def _build(self):
#         self._build_encoder()
#         self._build_decoder()
#         self._build_autoencoder()
#
#     def _build_autoencoder(self):
#         self._build_encoder()
#         self._build_decoder()
#         self.model = self.decoder(self.encoder)
#
#     def _build_decoder(self):
#         model = Sequential()
#         decoder_input = self._add_decoder_input(model=model)
#         dense_layer = self._add_dense_layer(model=decoder_input)
#         reshape_layer = self._add_reshape_layer(model=dense_layer)
#         conv_transpose_layers = self._add_conv_transpose_layers(model=reshape_layer)
#         self.decoder = self._add_decoder_output(model=conv_transpose_layers)
#
#     def _add_decoder_input(self, model):
#         """
#         The Sequential model before adding the decoder
#
#         :param (Sequential) model: The Sequential model before adding the input layer
#         :return: The Sequential model after adding the input layer
#         :rtype: Sequential
#         """
#         model.add(InputLayer(input_shape=self.latent_space_dim, name="decoder_input"))
#
#         return model
#
#     def _add_dense_layer(self, model):
#         """
#         ADd the dense layer to the decoder
#         :param (Sequential) model: The Sequential model before adding the dense model.
#         :return: The Sequential model after adding the dense model.
#         :rtype: Sequential
#         """
#         num_neurons = np.prod(self._shape_before_bottleneck)  # [1, 2, 4] -> 8
#         model.add(Dense(num_neurons, name="decoder_dense", activation=tf.nn.relu))
#         return model
#
#     def _add_reshape_layer(self, model):
#         """
#         Adding the reshaped model to the Sequential
#         :param (Sequential) model: The Sequential model before adding the reshaped layer.
#         :return: The Sequential model after the reshaped layer added.
#         :rtype: Sequential
#         """
#         model.add(Reshape(self._shape_before_bottleneck))
#         return model
#
#     def _add_conv_transpose_layers(self, model):
#         """
#         Add conv transpose blocks.
#
#         :param (Sequential) model: The Sequential model before adding the transpose conv layers
#         :return:The Sequential model aftering addding the transpose conv layers.
#         :rtype: Sequential
#         """
#         # loop through all the conv layers in reverse order and stop at the
#         # first layer
#
#         for layer_index in reversed(range(1, self._num_conv_layers)):
#             model = self._add_conv_transpose_layer(layer_index, model)
#         return model
#
#     def _add_conv_transpose_layer(self, layer_index, model):
#         """
#         Adding the a transpose conv layer to the Sequential model
#         :param  (int) layer_index:
#         :param (Sequential) model:
#         :return: The Sequential model aftering adding one transpose conv layer
#         :rtype: Sequential
#         """
#
#         layer_num = self._num_conv_layers - layer_index
#
#         model.add(Conv2DTranspose(
#             filters=self.conv_filters[layer_index],
#             kernel_size=self.conv_kernels[layer_index],
#             strides=self.conv_strides[layer_index],
#             padding="same",
#             name=f"decoder_conv_transpose_layer_{layer_num}"
#         ))
#
#         model.add(ReLU(name=f"decoder_relu_{layer_num}"))
#         model.add(BatchNormalization(name=f"decoder_bn_{layer_num}"))
#
#         return model
#
#     def _add_decoder_output(self, model):
#         """
#         Adding the decoder output layer
#         :param (Sequential) model: The Sequential model
#         :return: Return the Sequential model after adding the output layer
#         :rtype: Sequential
#         """
#
#         model.add(Conv2DTranspose(
#             filters=1,
#             kernel_size=self.conv_kernels[0],
#             strides=self.conv_strides[0],
#             padding="same",
#             name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
#         ))
#
#         model.add(Activation("sigmoid", name="sigmoid_layer"))
#
#         return model
#
#     def _build_encoder(self):
#         model = Sequential()
#         encoder_input = self._add_encoder_input(model)
#         conv_layers = self._add_conv_layers(encoder_input)
#         self.encoder = self._add_bottleneck(conv_layers)
#         # self._model_input = encoder_input
#         #  = Model(encoder_input, bottleneck, name="encoder")
#
#     def _add_encoder_input(self, model):
#         """
#
#         :param (Sequential) model: Sequential model
#         :return: The Sequential model with input layer added to it
#         """
#         # Input(shape=self.input_shape, name="encoder_input")
#         model.add(InputLayer(input_shape=self.input_shape, name="encoder_input"))
#         return model
#
#     def _add_conv_layers(self, encoder_input):
#         """
#         Create all convolutional blocks in encoder.
#
#         :param (Sequential) encoder_input: The Sequential model with input layer added to it
#         :return: The Sequential model with Conv layers added to it.
#         """
#         model = encoder_input
#         for layer_index in range(self._num_conv_layers):
#             model = self._add_conv_layer(layer_index, model)
#         return model
#
#     def _add_conv_layer(self, layer_index, model):
#         """
#         Add a convolutional block to a graph of layers, consisting of
#         conv 2d + ReLU + batch normalization.
#
#         :param (int) layer_index: Layer Index
#         :param (Sequential) model: The Sequential model
#         :return: The target Sequential model after adding the conv model.
#         """
#         layer_number = layer_index + 1
#
#         conv_layer = Conv2D(
#             filters=self.conv_filters[layer_index],
#             kernel_size=self.conv_kernels[layer_index],
#             strides=self.conv_strides[layer_index],
#             padding="same",
#             name=f"encoder_conv_layer_{layer_number}"
#         )
#         model.add(conv_layer)
#         model.add(ReLU(name=f"encoder_relu_{layer_number}"))
#         model.add(BatchNormalization(name=f"encoder_bn_{layer_number}"))
#
#         return model
#
#     def _add_bottleneck(self, model):
#         """
#         Flatten data and add bottleneck with Guassian sampling (Dense
#         layer).
#
#         :param (Sequential) model:
#         :return: The Sequential model after adding the bottleneck.
#         :rtype: Sequential
#         """
#
#         # Get the shape of the  latest layer of the Sequential model.
#         self._shape_before_bottleneck = model.layers[-1][1:]
#         # self._shape_before_bottleneck = K.int_shape(x)[1:]
#
#         model.add(Flatten)
#         # No activation
#         model.add(Dense(self.latent_space_dim + self.latent_space_dim))
#
#         # self.mu = Dense(self.latent_space_dim, name="mu")(x)
#         # self.log_variance = Dense(self.latent_space_dim,
#         #                           name="log_variance")(x)
#
#         return model
#
#         def sample_point_from_normal_distribution(args):
#             mu, log_variance = args
#             epsilon = K.random_normal(shape=K.shape(self.mu), mean=0.,
#                                       stddev=1.)
#             sampled_point = mu + K.exp(log_variance / 2) * epsilon
#             return sampled_point
#
#         x = Lambda(sample_point_from_normal_distribution,
#                    name="encoder_output")([self.mu, self.log_variance])
#         return x
#
#     @tf.function
#     def sample(self, eps=None):
#         if eps is None:
#             eps = tf.random.normal(shape=(100, self.latent_dim))
#         return self.decode(eps, apply_sigmoid=True)
#
#     def encode(self, x):
#         mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
#         return mean, logvar
#
#     def reparameterize(self, mean, logvar):
#         eps = tf.random.normal(shape=mean.shape)
#         return eps * tf.exp(logvar * .5) + mean
#
#     def decode(self, z, apply_sigmoid=False):
#         logits = self.decoder(z)
#         if apply_sigmoid:
#             probs = tf.sigmoid(logits)
#             return probs
#         return logits
#
#     def encode_predict(self, data):
#         """
#         Encode the data and generate the
#         :param data: Data to be encoded
#         :return: Encoded data
#         """
#         return self.encoder.predict(data)
#
#     def decode_predict(self, data):
#         """
#         Decode the data
#
#         :param data: Data to be decoded
#         :return: Reconstructed data
#         """
#         return self.decoder.predict(data)
#
#     def log_normal_pdf(self, sample, mean, logvar, raxis=1):
#         log2pi = tf.math.log(2. * np.pi)
#         return tf.reduce_sum(
#             -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
#             axis=raxis)
#
#     def compute_loss(self, x):
#         mean, logvar = self.encode(x)
#         z = self.reparameterize(mean, logvar)
#         x_logit = self.decode(z)
#         cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
#         logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
#         logpz = self.log_normal_pdf(z, 0., 0.)
#         logqz_x = self.log_normal_pdf(z, mean, logvar)
#         return -tf.reduce_mean(logpx_z + logpz - logqz_x)
