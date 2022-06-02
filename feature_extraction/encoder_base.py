from feature_extraction import FeatureExtractor


class EncoderBaseFeatures(FeatureExtractor):
    def __init__(self,
                 to_read_dir_path,
                 to_store_dir_path,
                 encoder_models_dir_path,
                 encoder_model_weight_file_name,
                 encoder_model_parameters_file_name,
                 latent_space_dim
                 ):
        """

        Extract the features based on the auto encoder by loading the model

        :param (str) to_read_dir_path: Directory path to read the files
        :param (str) to_store_dir_path: Directory path to store the features
        :param (str) encoder_models_dir_path: Auto encoder model directory path
        :param (str) encoder_model_weight_file_name: Auto encoder model weight file name
        :param (str) encoder_model_parameters_file_name: Auto encoder model parameters file name
        :param (int) latent_space_dim: Latent space dimension
        :returns: None
        """

        super().__init__(to_read_dir_path, to_store_dir_path)

        self.encoder_models_dir_path = encoder_models_dir_path
        self.encoder_model_weight_file_name = encoder_model_weight_file_name
        self.encoder_model_parameters_file_name = encoder_model_parameters_file_name
        self.latent_space_dim = latent_space_dim

    def _loader(self, file_path):
        raise NotImplemented("This method has not implemented yet")

    def process(self):
        raise NotImplemented("This method has not implemented yet")

    def _save_feature(self, feature, file_path):
        raise NotImplemented("This method has not implemented yet")
