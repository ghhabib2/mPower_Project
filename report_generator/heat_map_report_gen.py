import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from report_generator import ReportABCCLass


class TuningReportGenerator(ReportABCCLass):

    def __init__(self, to_read_dir_path, metric_to_check, latent_dim=2):

        """

        Generate the heatmap for the data

        :param (str) to_read_dir_path: Path to the root folder of the model
        :param (int) latent_dim: Latent space dimension.
        :param (str) metric_to_check: The metric to consider for the process
        :return:
        """

        self._latent_dim = latent_dim
        self._metric_to_check = metric_to_check
        self._list_data = []
        self._data_frame = None

        super().__init__(to_read_dir_path)

    def load(self):
        # Load the list of the directories
        dir_list = os.listdir(self._TO_READ_PATH)

        for item in dir_list:
            if item != ".DS_Store":
                train_dir_path = os.path.join(self._TO_READ_PATH, item)
                file_path = os.path.join(train_dir_path, "trial.json")
                if not os.path.exists(file_path):
                    continue
                with open(file_path) as f:
                    jason_file = json.load(f)
                    best_latent_space_dim = pd.DataFrame.from_dict(
                        pd.DataFrame.from_dict(
                            jason_file
                        ).iloc[1]).iloc[1]['values']['best_latent_space_dim']
                    if best_latent_space_dim != self._latent_dim:
                        continue
                    best_conv_filter = pd.DataFrame.from_dict(
                        pd.DataFrame.from_dict(
                            jason_file
                        ).iloc[1]).iloc[1]['values']['best_conv_filter']
                    best_kernel_size = pd.DataFrame.from_dict(
                        pd.DataFrame.from_dict(
                            jason_file
                        ).iloc[1]).iloc[1]['values']['best_kernel_size']
                    metric_value = pd.DataFrame.from_dict(
                        pd.DataFrame.from_dict(
                            jason_file
                        ).iloc[2]).iloc[2][0][self._metric_to_check]['observations'][0]['value'][0]
                    self._list_data.append([best_conv_filter, best_kernel_size, metric_value])

        # Convert data to DataFrame
        self._data_frame = pd.DataFrame(data=self._list_data, columns=['conv', 'kernel', 'value'])

    def plot(self):
        conv = pd.DataFrame.from_dict(
            self._data_frame['conv']).drop_duplicates().sort_values(by='conv', ascending=True).to_numpy().flatten()
        kernel = pd.DataFrame.from_dict(
            self._data_frame['kernel']).drop_duplicates().sort_values('kernel', ascending=True).to_numpy().flatten()

        values = np.zeros((len(conv), len(kernel)))

        for index1, i in enumerate(conv):
            for index2, j in enumerate(kernel):
                filter_data = self._data_frame[
                    (self._data_frame['conv'] == i) & (self._data_frame['kernel'] == j)]
                if len(filter_data) == 0:
                    values[index1, index2] = 0
                else:
                    values[index1, index2] = filter_data.iloc[0, 2]

        fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
        im = ax.imshow(values, aspect='auto')

        # Show all ticks and label them with the respective list entries
        ax.set_yticks(np.arange(len(conv)), labels=conv.astype(str))
        ax.set_xticks(np.arange(len(kernel)), labels=kernel.astype(str))

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(conv)):
            for j in range(len(kernel)):
                if values[i, j] == 0:
                    text = ax.text(j, i, "---",
                                   ha="center", va="center", color="w")
                else:
                    text = ax.text(j, i, "{:.3f}".format(values[i, j]),
                                   ha="center", va="center", color="w")

        ax.set_title(f"Heat Map of tuning result for latent space with dimension equals to {self._latent_dim}")
        fig.tight_layout()
        plt.show()
