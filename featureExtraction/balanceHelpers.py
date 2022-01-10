import numpy as np
import json
import os
import math
from datetime import timedelta, datetime
import functools
from statistics import mean, median
from scipy.stats import median_absolute_deviation, pearsonr
import statsmodels.api as sm
import pandas as pd

# Tapping Helper Function
ROOT_URL = os.path.join(os.getcwd(), "collected_data")

# Emulate Foldl function
foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)


def tested_jason():
    file_path = os.path.join(ROOT_URL,
                             "deviceMotion_walking_outbound.json.items-62cd8d92-58a4-4c6e-b0cb-5783fa7befc57597078404659349876.tmp")

    with open(file_path) as f:
        data = json.load(f)
        shape_balance_data(data)


def shape_balance_data(data):
    """
    Preprocess the accelrometer data

    :param data: Raw data to be processed
    :type data: list
    :return: Shaped data to be processed for feature extraction
    :rtype: list
    """

    def time_stamp_adder(o_list: list, item):
        o_list.append(item['timestamp'])
        return o_list

    def accel_adder(o_list: list, item):
        o_list.append([item['userAcceleration']['x'],
                       item['userAcceleration']['y'],
                       item['userAcceleration']['z']])
        return o_list

    time_stamp = foldl(time_stamp_adder, [], data)

    primary_time_stamp = time_stamp[0]

    # Calculate the delta times based on the data recorded
    time_stamp = list(np.array(time_stamp) - primary_time_stamp)

    user_acceleration = np.array(foldl(accel_adder, [], data))

    # merge the seperated records into one list
    # Return the data for further process.
    return list(zip(time_stamp, user_acceleration[:, 0], user_acceleration[:, 1], user_acceleration[:, 2]))

