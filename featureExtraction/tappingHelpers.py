import numpy as np
import json
import os
import utils.stat as stat
import functools
from statistics import mean, median

# Tapping Helper Function
ROOT_URL = os.path.join(os.getcwd(), "collected_data")

# Emulate Foldl function
foldl = lambda func, acc, xs: functools.reduce(func, xs, acc)


def tested_jason():
    file_path = os.path.join(ROOT_URL,
                             "tapping_results.json.TappingSamples-41126e25-aa15-419f-af14-1aa8e78d870e1383831155599447443.tmp")

    with open(file_path) as f:
        data = json.load(f)
        shaped_unfiltered = ShapeTappingData(data)
        shaped_filtered = CleanTappedButtonNone(shaped_unfiltered)
        feature_vector = extracttapping(shaped_filtered, shaped_unfiltered)
        print(feature_vector)


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
    :type data: list
    :return: Return a list of tuples in the following format [(time, buttonid, coordination_x, coordination_y)]
    :rtype: list
    """

    # List that oi
    output_list = []
    for item in data:
        time = item['TapTimeStamp']
        button_id = item['TappedButtonId']
        coord = get_xy((item['TapCoordinate']))
        output_list.append((time, button_id, coord[0], coord[1]))

    return output_list


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

    def adder(o_list: list, data_record: tuple):
        if data_record[1] != "TappedButtonNone":
            o_list.append(data_record)
        return o_list

    # Filter out the `TappedButtonNone` from the list of data
    output_list = foldl(adder, [], input_list)

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

    def time_adder(o_list: list, x: tuple):
        o_list.append(x[0])
        return o_list

    def x_adder(o_list: list, x: tuple):
        o_list.append(x[2])
        return o_list

    # Extract the list of recorded times
    time_list = foldl(time_adder, [], data)
    tap_t = [time - time_list[0] for time in time_list]

    # Find left/right finger "depress" event
    x_list = foldl(x_adder, [], data)
    x_list = np.array(x_list) - np.mean(x_list)
    d_x = np.diff(x_list).tolist()

    condition_func = lambda temp_d_x: np.abs(temp_d_x) > depressThr
    bool_d_x = condition_func(d_x)

    # Filter data
    filtered_list_indexs = np.where(bool_d_x)[0]
    dat = [data[item] for item in filtered_list_indexs]
    tap_t = [tap_t[item] for item in filtered_list_indexs]

    # Find depress event intervals
    tap_inter = np.diff(tap_t).tolist()

    # Return the result as tuple of two lists
    return dat, tap_inter





def extracttapping(tapping_data_list, tapping_data_list_non_cleaned):
    """
    Extract the Tapping features

    :param tapping_data_list: List of tuples ready for tapping extraction
    :type tapping_data_list: list
    :param tapping_data_list_non_cleaned: Uncleaned tapping informatoin
    :type   tapping_data_list_non_cleaned: list
    :return: Return a dictionary of feature extracted
    :rtype: dict
    """

    aux = GetLeftRightEventsAndTapIntervals(tapping_data_list, depressThr=20)
    tap_inter = aux[1]
    dat = aux[0]

    if len(dat) == 0:
        return None

    # Extract the x coordination data from the shaped and cleaned list of tuples
    def x_adder(o_list: list, x: tuple):
        o_list.append(x[2])
        return o_list

    def y_adder(o_list: list, x: tuple):
        o_list.append(x[3])
        return o_list

    x = foldl(x_adder, [], dat)
    y = foldl(y_adder, [], dat)

    mean_x = mean(x)

    condition_func = lambda temp_d_x: np.array(x) < mean_x
    bool_x_arr = condition_func(x)
    i_l = np.where(bool_x_arr)[0]

    condition_func = lambda temp_d_x: np.array(x) >= mean_x
    bool_x_arr = condition_func(x)
    i_r = np.where(bool_x_arr)[0]

    drift_left = stat.drift([x[item] for item in i_l], [y[item] for item in i_l])
    drift_right = stat.drift([x[item] for item in i_r], [y[item] for item in i_r])

    try:
        aux_acf = stat.acf(tap_inter)
    except ArithmeticError:
        aux_acf = (None, None, None)

    try:
        aux_fatigue = stat.fatigue(tap_inter)
    except ArithmeticError:
        aux_fatigue = (None, None, None)

    # TODO Adding the proper implementation of DFA
    # TODO Adding the DFA related features

    # iqr function for 1-D array
    def iqrFunc(x):
        iqr = lambda x: np.percentile(x, [75, 25])
        q3, q1 = iqr(tap_inter)
        return q3 - q1

    def buttonNonFreqCalcFunc(x):

        def adder(o_list: list, data_record: tuple):
            if data_record[1] == "TappedButtonNone":
                o_list.append(data_record)
            return o_list

        non_button_list = foldl(adder, [], x)

        return len(non_button_list) / len(x)

    return {'meanTapInter': mean(tap_inter),
            'medianTapInter': median(tap_inter),
            'iqrTapInter': iqrFunc(tap_inter),
            'minTapInter': min(tap_inter),
            'maxTapInter': max(tap_inter),
            'skewTapInter': stat.kewness(tap_inter),
            'kurTapInter': stat.kurtosis(tap_inter),
            'sdTapInter': np.std(tap_inter),
            'madTapInter': stat.mad(tap_inter),
            'cvTapInter': stat.cv(tap_inter),
            'rangeTapInter': (np.diff([min(tap_inter), max(tap_inter)])),
            'tekoTapInter': stat.mean_tkeo(tap_inter),
            'ar1TapInter': aux_acf[1],
            'ar2TapInter': aux_acf[2],
            'fatigue10TapInter': aux_fatigue[0],
            'fatigue25TapInter': aux_fatigue[1],
            'fatigue50TapInter': aux_fatigue[2],
            'meanDriftLeft': mean(drift_left),
            'medianDriftLeft': median(drift_left),
            'iqrDriftLeft': iqrFunc(drift_left),
            'minDriftLeft': min(drift_left),
            'maxDriftLeft': max(drift_left),
            'skewDriftLeft': stat.skewness(drift_left),
            'kurDriftLeft': stat.kurtosis(drift_left),
            'sdDriftLeft': np.std(drift_left),
            'madDriftLeft': stat.mad(drift_left),
            'cvDriftLeft': stat.cv(drift_left),
            'rangeDriftLeft': (np.diff([min(drift_left), max(drift_left)])),
            'meanDriftRight': mean(drift_right),
            'medianDriftRight': median(drift_right),
            'iqrDriftRight': iqrFunc(drift_right),
            'minDriftRight': min(drift_right),
            'maxDriftRight': max(drift_right),
            'skewDriftRight': stat.skewness(drift_right),
            'kurDriftRight': stat.kurtosis(drift_right),
            'sdDriftRight': np.std(drift_right),
            'madDriftRight': stat.mad(drift_right),
            'cvDriftRight': stat.cv(drift_right),
            'rangeDriftRight': (np.diff([min(drift_right), max(drift_right)])),
            'numberTaps': len(dat),
            'buttonNonFreq': buttonNonFreqCalcFunc(tapping_data_list_non_cleaned),
            'corXY': stat.cor(x, y)
            }
