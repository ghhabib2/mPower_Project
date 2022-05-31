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
                             "MemoryGameResults.json.MemoryGameGameRecords-9b24ce2e-b1a0-48a6-b121-1c116de1397d2885138818597878392.tmp")

    with open(file_path) as f:
        data = json.load(f)
        print(memory_game_generate_summary_stats(data))


def processCoordLine(data):
    """

    :param data: Coordination that user clicked
    :type data: str
    :return: Tuple of user clicked coordination
    :rtype: (float, float)
    """

    splitted_data = data.split(",")

    x_coord = float(splitted_data[0][1:])
    y_coord = float(splitted_data[1][:len(splitted_data[1]) - 1])
    return x_coord, y_coord


def get_flower_centers(targeted_rects):
    """
    Return the flowers center points based on the positions and sizes of the flowers.

    :param targeted_rects: Positions and sizes of the flowers
    :type targeted_rects: list

    :return: list of information for flowers center and coordination.
    :rtype: list
    """

    # List for storing the output sizes and coordination
    out_put_list = []

    for item in targeted_rects:
        coordinations_sizes = str(item).split(",")
        x_cord = int(coordinations_sizes[0][2:])
        y_cord = int(coordinations_sizes[1][:len(coordinations_sizes[1]) - 1])
        width = int(coordinations_sizes[2][2:])
        height = int(coordinations_sizes[3][:len(coordinations_sizes[3]) - 2])

        x_midpoint = x_cord + width / 2
        y_midpoint = y_cord + height / 2
        # Adding the coordination for the flower to the output list
        out_put_list.append([x_cord, y_cord, width, height, x_midpoint, y_midpoint])

    # Return the output list
    return out_put_list


def process_subseq_in_a_game(data, targeted_rects):
    """
    Have MemoryGameRecordTouchSamples as input and return the distances and delta time based on how user played the
    memory game

    :param data: List of MemoryGameRecordTouchSamples
    :type data: list
    :param targeted_rects: Targeted Rects which shows the followers positions and size of the followers.
    :type targeted_rects: list
    :return: Tuple of two lists. First list is based on the distances calculated based on user performance and the
            second is the delta time based on how much user fast in terms of touching the screen.
    :rtype: (list, list, list)
    """

    # Get the follower's center positions and coordination's
    flower_centers = get_flower_centers(targeted_rects)
    distances = []
    time_deltas = []
    touch_sample_status = []
    one_off_time_stamp = float(data[0]["MemoryGameTouchSampleTimestamp"]) * 1000
    # Go over the data of user touch smaples
    for item in data:
        user_x_coord, user_y_coord = processCoordLine(item["MemoryGameTouchSampleLocation"])
        target_index = int(item["MemoryGameTouchSampleTargetIndex"]) + 1
        x_midpoint = flower_centers[target_index - 1][4]
        y_midpoint = flower_centers[target_index - 1][5]
        distances.append(math.sqrt((x_midpoint - user_x_coord) ** 2 + (y_midpoint - user_y_coord) ** 2))
        time_deltas.append(float(item["MemoryGameTouchSampleTimestamp"]) * 1000 - one_off_time_stamp)
        touch_sample_status.append(bool(item["MemoryGameTouchSampleIsCorrect"]))
    return distances, time_deltas, touch_sample_status


def process_game(data):
    """
    Receive the raw memory game data as input and return the processed data for feature extraction as output

    :param data: Raw input of memory game data
    :type data: list
    :return: Return the processed data for feature extraction as list
    :rtype: list
    """
    # Output list
    out_put_list = []
    # Go over the information of each game session
    for item in data:
        flower_matrix_size = int(item["MemoryGameRecordGameSize"])
        sequence = list(np.array(item["MemoryGameRecordSequence"]) + 1)
        game_size = int(len(sequence))
        distances, time_deltas, touch_sample_status = process_subseq_in_a_game(item["MemoryGameRecordTouchSamples"],
                                                                               item["MemoryGameRecordTargetRects"])
        total_distance = sum(distances)
        total_time_delta = sum(time_deltas)
        total_correct = foldl((lambda y, item: y + 1 if item else y), 0, touch_sample_status)
        total_not_correct = len(touch_sample_status) - total_correct
        game_status = bool(item["MemoryGameStatus"])
        # new_flower_touched = len(item["MemoryGameRecordTouchSamples"]) - len(item["MemoryGameRecordTargetRects"])
        # TODO Think about how to find the new touched flower, for now pass it as zero
        new_flower_touched = 0
        out_put_list.append([flower_matrix_size, game_size, sequence, distances, time_deltas, touch_sample_status,
                             total_distance,
                             total_time_delta,
                             total_correct,
                             total_not_correct,
                             game_status,
                             new_flower_touched])

    # return the processed raw data for feature extraction
    return out_put_list


def memory_game_generate_summary_stats(unprocessed_data):
    """
    Receive the processed data as input and generate output features

    :param unprocessed_data: Processed data ready for feature extraction
    :type unprocessed_data: list
    :return: Return the feature dictionary
    :rtype: dict
    """

    try:

        data = process_game(unprocessed_data)

        def distance_adder(o_list: list, item):
            o_list.append(item[6])
            return o_list

        def time_delta_adder(o_list: list, item):
            o_list.append(item[7])
            return o_list

        def correct_adder(o_list: list, item):
            o_list.append(item[8])
            return o_list

        def not_correct_adder(o_list: list, item):
            o_list.append(item[9])
            return o_list

        def total_new_flower_adder(o_list: list, item):
            o_list.append(item[11])
            return o_list

        distances = foldl(lambda o_list, item: distance_adder(o_list, item), [], data)
        time_deltas = foldl(lambda o_list, item: time_delta_adder(o_list, item), [], data)
        total_distance =  sum(distances)
        total_time = sum(time_deltas)
        total_correct_flowers = sum(foldl(lambda o_list, item: correct_adder(o_list, item), [], data))
        avg_wrong_flower_num = mean(foldl(lambda o_list, item: not_correct_adder(o_list, item), [], data))
        total_new_flowers_touched = sum(foldl(lambda o_list, item: total_new_flower_adder(o_list, item), [], data))

        var_time = np.var(total_time)
        mean_time = np.mean(total_time)
        med_time = np.median(total_time)
        med_dist = np.median(total_distance)
        mean_dist = np.mean(total_distance)
        var_dist = np.var(total_distance)

        return {"totalDistance": total_distance,
                "totalTime": total_time,
                "totalCorrectFlowers": total_correct_flowers,
                "avg_wrongflowerNum" : avg_wrong_flower_num,
                "total_newFlowers_touched": total_new_flowers_touched,
                "varTime" : var_time,
                "meanTime": mean_time,
                "medTime": med_time,
                "medDist": med_dist,
                "meanDist": mean_dist,
                "varDist": var_dist,
                "error": False
                }
    except ValueError as ex:
        print(ex)
        return {"totalDistance": 0,
                "totalTime": 0,
                "totalCorrectFlowers": 0,
                "avg_wrongflowerNum": 0,
                "total_newFlowers_touched": 0,
                "varTime": 0,
                "meanTime": 0,
                "medTime": 0,
                "medDist": 0,
                "meanDist": 0,
                "varDist": 0,
                "error": True
                }
