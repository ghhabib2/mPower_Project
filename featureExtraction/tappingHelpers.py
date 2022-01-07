import numpy as np
import json
import os
import functools
from statistics import mean
import statsmodels.api as sm
import pandas as pd
import operator

# Tapping Helper Function
ROOT_URL = os.path.join(os.getcwd(), "collected_data")

# Emulate Foldl function
foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)


def tested_jason():
    file_path = os.path.join(ROOT_URL
                             ,
                             "tapping_results.json.TappingSamples-41126e25-aa15-419f-af14-1aa8e78d870e1383831155599447443.tmp")

    with open(file_path) as f:
        data = json.load(f)
        for i in data:
            print(i['TapTimeStamp'], "\n")


def get_xy(coordination_str):
    """
    Extract the coordination from the data

    :param coordination_str:
    :type coordination_str: str
    :return: Extract the coordination from the data
    :rtype: (float,float)
    """

    # Remove the brackets from the coordination string
    coordination_str = coordination_str[1:len(coordination_str) - 1].split(",")
    # Return the coordination values as a tuple
    return float(coordination_str[0]), float(coordination_str[1])


def ShapeTappingData(data):
    """
    Extract the tapping information from each recorded samples in the file

    :param data: Tapping data sample
    :type data: dict
    :return: Return the tuple of data holding this information (time, buttonid, tap_coordination)
    :rtype: (float, str, float, float)
    """

    time = data['TapTimeStamp']
    button_id = data['TappedButtonId']
    coord = get_xy((data['TapCoordinate']))

    return time, button_id, coord[0], coord[1]


def CleanTappedButtonNone(input_list):
    """
    Get list of tuples as input data, and filter out the `TappedButtonNone` from the list.
    Each element in the list is a tuple in the following format: (time:float, button_id:str, coord, tuple). Each
    element of the output also has the same shape and type.
    
    :param input_list: Input list of tuples
    :type input_list: list
    :return: Output filtered list of tuples
    :rtype: list 
    """

    # Filter out the `TappedButtonNone` from the list of data
    output_list = foldl(
        lambda o_list, data_record: o_list.append(data_record) if data_record[1] is not "TappedButtonNone" else o_list,
        [], input_list)

    # Sort the list of tuples based on the filtered output and timestamp
    return sorted(output_list, key=lambda x: x[0])


def GetLeftRightEventsAndTapIntervals(data: list, depressThr=20):
    """
    Computes tapping time series
    Tapping interval and tapping position

    :param data: Input list of tuples for interval calculation.
    :type data: list
    :param depressThr:
    :return: Return the Computed tapping time series
    :rtype: (list, list)
    """

    # Extract the list of recorded times
    time_list = foldl(lambda t_temp_list, x: t_temp_list.append(x[0]), [], data)
    tap_t = [time - time_list[0] for time in time_list]

    # Find left/right finger "depress" event
    x_list = foldl(lambda x_temp_list, x: x_temp_list.append(x[2]), [], data)
    x_list = x_list - mean(x_list)
    d_x = np.diff(x_list).tolist()
    condition_func = lambda temp_d_x: np.array(temp_d_x) > depressThr
    bool_d_x = condition_func(d_x)

    # Filter data
    filtered_list_indexs = np.where(bool_d_x)[0]
    dat = [data[item] for item in filtered_list_indexs]
    tap_t = [tap_t[item] for item in filtered_list_indexs]

    # Find depress event intervals
    tap_inter = np.diff(tap_t).tolist()

    # Return the result as tuple of two lists
    return dat, tap_inter


def cv(x):
    """
    :param x: A list
    :type x: list
    :return:
    """
    if len(x) < 3:
        return None
    else:
        return np.std(x) / np.mean(x) * 100


def mean_tkeo(x):
    if len(x) < 3:
        return None
    else:
        y = np.power(x, 2) - np.array(x[1:]) * np.array(x[0:len(x) - 1])
        return np.mean(y)


def fatigue(x):
    x_length = len(x)
    if x_length < 3:
        return None, None, None
    else:
        top10 = round(0.1 * x_length)
        top25 = round(0.25 * x_length)
        top50 = round(0.5 * x_length)

        fatigue10 = np.mean(x[0:top10]) - np.mean(x[x_length - top10:])
        fatigue25 = np.mean(x[0:top25]) - np.mean(x[x_length - top25:])
        fatigue50 = np.mean(x[0:top50]) - np.mean(x[x_length - top50:])

        return fatigue10, fatigue25, fatigue50


def skewness(x):
    if len(x) < 3:
        return None
    else:
        mu = np.mean(x)
        return np.power(np.mean(np.array(x) - mu), 3) / np.power(np.mean(np.power(np.array(x) - mu, 2)), 3 / 2)


def kurtosis(x):
    if len(x) < 3:
        return None
    else:
        mu = np.mean(x)
        return np.mean(np.power(np.mean(np.array(x) - mu), 4)) / np.power(np.mean(np.power(np.array(x) - mu, 2)), 2)


def acf(x):
    if len(x) < 3:
        return None, None, None
    else:
        sm.tsa.acf(x, nlags=2)


def drift(x: list, y: list):
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))

