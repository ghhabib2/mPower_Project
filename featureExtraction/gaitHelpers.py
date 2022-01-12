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
        o_list.append([float(item['userAcceleration']['x']),
                       float(item['userAcceleration']['y']),
                       float(item['userAcceleration']['z'])])
        return o_list

    time_stamp = foldl(time_stamp_adder, [], data)

    primary_time_stamp = time_stamp[0]

    # Calculate the delta times based on the data recorded
    time_stamp = list(np.array(time_stamp) - primary_time_stamp)

    user_acceleration = np.array(foldl(accel_adder, [], data))

    # merge the seperated records into one list
    # Return the data for further process.
    return list(zip(time_stamp, user_acceleration[:, 0], user_acceleration[:, 1], user_acceleration[:, 2]))


def single_axis_features(data, time):
    """
    Calculate the features for single axis

    :param data: Data of the axis as list
    :type data: list
    :param time: Time list based on the information shaped
    :type time: list
    :return: dictionary with features extracted for the axis.
    :rtype: dict
    """

    mean_x = np.mean(data)
    sd_x = np.std(data)
    mod_x = stat.mode(data)
    skew_x = stat.skewness(data)
    kur_x = stat.kurtosis(data)
    aux_x = np.quantile(data, q=[0, 0.25, 0.5, 0.75, 1])
    q1_x = aux_x[1]
    median_x = aux_x[2]
    q3_x = aux_x[3]
    iqrx = q3_x - q1_x
    range_x = aux_x[4] - aux_x[0]
    acf_x = stat.acf(data, n_lags=1)
    zcr_x = stat.zcr(data)
    # TODO find an stable way for calculation of DFA. Fow now pass the value as np.nan
    dfa_x = np.nan
    cv_x = stat.cv(data)
    tkeo_x = stat.mean_tkeo(data)
    frequency, power = stat.lsp(data, time)
    p0_x = max(power)
    f0_x = frequency[list(power).index(p0_x)]
    # Return the extracted features as a dictionary
    return {"mean": mean_x,
            "sd": sd_x,
            "mode": mod_x,
            "skew": skew_x,
            "kur": kur_x,
            "q1": q1_x,
            "median": median_x,
            "q3": q3_x,
            "iqr": iqrx,
            "range": range_x,
            "acf": acf_x,
            "zcr": zcr_x,
            "dfa": dfa_x,
            "cv": cv_x,
            "tkeo": tkeo_x,
            "F0X": f0_x,
            "P0X": p0_x
            }


def accel_low_pass_filter(data, alpha):
    """
    Apply low pass filter on information of motion sensor
    :param data: Data to be processed
    :type data: list
    :param alpha: Alpha value to by applied in LPF
    :type alpha : float
    :return: Return the data after applying the LFP
    :rtype : list
    """
    # Convert list to numpy array
    dat = np.array(data)

    n = len(data)
    a_x = dat[:, 1]
    a_y = dat[:, 2]
    a_z = dat[:, 3]

    for i in range(1, n):
        a_x[i] = alpha * a_x[i] + (1 - alpha) * a_x[i - 1]
        a_y[i] = alpha * a_y[i] + (1 - alpha) * a_y[i - 1]
        a_z[i] = alpha * a_z[i] + (1 - alpha) * a_z[i - 1]

    dat[:, 1] = a_x
    dat[:, 2] = a_y
    dat[:, 3] = a_z

    return list(dat)


def get_gait_features(data, alpha=1):
    """
    Extract the features based on the shaped data

    :param data: Data to be processed for feature extraction
    :type data: list
    :param alpha: Alpha value for LFP process. Default value is 1 unless changed by the user
    :type alpha: float
    :return:
    """

    # Apply the LFP while converting to numpy array for convince of addressing and computation
    dat = np.array(accel_low_pass_filter(data, alpha))

    x = dat[:, 1]
    y = dat[:, 2]
    z = dat[:, 3]

    aa = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    aj = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    t = dat[:, 0]

    #############################
    out_x = single_axis_features(x, t)
    out_y = single_axis_features(y, t)
    out_z = single_axis_features(z, t)
    out_aa = single_axis_features(aa, t)
    out_aj = single_axis_features(aj, list(np.array(t)[1:]))
    #############################
    cor_xy = stat.cor(x, y)
    cor_xz = stat.cor(x, z)
    cor_yz = stat.cor(y, z)
    cors = [cor_xy, cor_xz, cor_yz]

    return {
        "outX": out_x,
        "outY": out_y,
        "outZ": out_z,
        "outAA": out_aa,
        "outAJ": out_aj,
        "cors": cors,
    }
