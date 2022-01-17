import numpy as np
import json
import os
import utils.stat as stat
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
        data = shape_balance_data(data)
        print(get_gait_features(data))


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
        o_list.append([float(item['userAcceleration']['x'] * 9.8),
                       float(item['userAcceleration']['y'] * 9.8),
                       float(item['userAcceleration']['z'] * 9.8)])
        return o_list

    time_stamp = foldl(time_stamp_adder, [], data)

    primary_time_stamp = time_stamp[0]

    # Calculate the delta times based on the data recorded
    time_stamp = list(np.array(time_stamp) - primary_time_stamp)

    user_acceleration = np.array(foldl(accel_adder, [], data))

    # merge the seperated records into one list
    # Return the data for further process.
    return list(zip(time_stamp, user_acceleration[:, 0], user_acceleration[:, 1], user_acceleration[:, 2]))


def trim_data(data, time_start=5, time_end=None):
    """
    Function for Timing the Data based on the start and end time limitations.
    :param data: Data to be trimmed
    :type data: list
    :param time_start: Start time
    :type time_start : float
    :param time_end: End time
    :type time_end : float
    :return: Return the trimmed list of data
    :rtype: np.ndarray
    """

    # Separate the time stamp column
    time = np.array(data)[:, 0]

    if time_end is None:
        time_end = time[len(time)]

    # Calculate the starting and ending index
    start_index = np.where(time == min(time[time >= time_start]))[0]
    end_index = np.where(time == max(time[time <= time_start]))[0]

    # trim the data based on the selected starting and ending indexes
    trimmed_data = np.array(data)[start_index:end_index, :]

    # Find the new time stamps based on the newly calculated interval.
    trimmed_data[0] = trimmed_data[:, 0] - trimmed_data[0, 0]

    # Return the trimmed and reshaped data
    return trimmed_data


def get_balance_features(data, time_start=5, time_end=None):
    data = trim_data(data, time_start, time_end)

    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]

    aa = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    mean_aa = np.mean(aa)
    sd_aa = np.std(aa)
    mode_aa = stat.mode(aa)
    skew_aa = stat.skewness(aa)
    kur_aa = stat.kurtosis(aa)
    iqr_aa, median_aa, range_aa = stat.iqr_median_range_calculator(aa)
    acf_aa = stat.acf(aa, n_lags=1)
    zcr_aa = stat.zcr(aa)

    # TODO Find a proper way for calculation of dfa for now pass the value of dfa as none
    dfa_aa = np.nan

    bpa = feature_bpa(data)


# TODO find out what is BPA?
def feature_bpa(post):
    """
    Calculate the BPA features

    :param post: Data for feature extractoin
    :type post: np.ndarray

    :return: Tuple of features in the following format (Maximum force, Average scaled power X, Y, Z,
     Detrended fluctuation analysis)
    :rtype: tuple
    """

    time = post[:, 0] - post[0, 0]

    d_time = time[len(time)] - time[0]

    post = post[1:, :]

    n = len(post)

    # Calculate the Orientation
    mg = post.mean(axis=0)
    # Orientation-corrected force signals
    post_force = post - np.tile(mg, (n, 1))

    dt = np.diff(time)
    dt = np.concatenate((dt, [dt[len(dt)]]))

    # np.transpose([an_array] * repetitions)
    post_vel = np.cumsum((post_force * np.transpose([dt] * 3)))

    # Average scaled power X, Y, Z
    post_power = np.mean((0.5 * 70 * post_vel ** 2).sum(axis=0) / d_time) / 1e4

    # Force vector magnitude signal
    post_mag = np.sqrt((post_force ** 2).sum(axis=1))

    # Maximum force
    post_peak = np.quantile(post_mag, [0.95]) / 10

    # Detrended fluctuation analysis scaling exponent
    # TODO fluctuation analysis using the DFA. Since we do not have DFA yet. This will be postpoined.
    alpha = np.nan
    return post_peak, post_power, alpha


def get_displacement(time, accel):
    """
    Find the displacement based on the input

    :param time: Time data
    :type time: list
    :param accel: Accelerometer data
    :type accel: np.ndarray
    :return: tuple of vel and dis
    :rtype: tuple
    """
    delta_time = np.diff(time)
    n = len(delta_time)
    vel = []
    dis = []

    vel.append(0)
    dis.append(0)

    for i in range(start=1, stop=n):
        vel.append(vel[i - 1] + 0.5 * (accel[i] + accel[i - 1]) * delta_time[i])
        dis.append(dis[i - 1] + 0.5 * (vel[i] + vel[i - 1]) * delta_time[i])

    return np.array(vel), np.array(dis)


def get_xyz_displacement(x):
    """
    Calculate the displacement

    :param x: The data to be analysed
    :type x: np.ndarray
    :return: Return the tuple of displacement for each of the axis
    :rtype: tuple
    """
    dist_x = get_displacement(x[:, 0], x[:, 1])
    dist_y = get_displacement(x[:, 0], x[:, 2])
    dist_z = get_displacement(x[:, 0], x[:, 3])

    return dist_x, dist_y, dist_z


def center_acceleratoin(x):
    data = np.array(x)
    data[:, 1] = data[:, 1] - np.median(data[:, 1])
    data[:, 2] = data[:, 2] - np.median(data[:, 2])
    data[:, 3] = data[:, 3] - np.median(data[:, 3])
    return data


def box_volume_feature(data):
    """
    :param data: Data to be analysed
    :return:
    """

    data = center_acceleratoin(data)
    au_x = get_xyz_displacement(data)

    # TODO do the feature calculation based on the following link
    # https://github.com/ghhabib2/mPower-sdata/edit/master/featureExtraction/balanceHelpers.R
