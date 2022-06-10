# Define the libraries Roots
import os
import numpy as np
# import matlab.engine
import soundfile as sf
import librosa
import time
import gc

import parselmouth
from parselmouth.praat import call


import statistics
import speechpy
# import emd
import math
import itertools

BASE_DIR = os.getcwd()
MATLAB_ROOT = os.path.join(BASE_DIR, "matlab_root")
MEDIA_ROOT = os.path.join(BASE_DIR, "temp_media_folder")
VOICE_BOX = os.path.join(MATLAB_ROOT, "voicebox/voicebox")
VOICE_ANALYSIS_TOOLBOX = os.path.join(MATLAB_ROOT, "VoiceAnalysisToolbox")
DFA = os.path.join(MATLAB_ROOT, "dfa")
EMD = os.path.join(MATLAB_ROOT, "emd")
RPDE = os.path.join(MATLAB_ROOT, "rpde")
SHRP = os.path.join(MATLAB_ROOT, "shrp")


def matlab_base_feature_downloader(file_path, segment_duration=2):
    """
    Download the mfcc features based on the information presented in data file

    the data file is going to keep the `file_name` and `user_id` and `gender`

    :param file_path Audio file path.
    :type file_path str
    :param segment_duration Segment duration in seconds
    :type segment_duration int
    :return: Return the tuples of three elements storing the features name and values based on the number of selected
    segments.
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """

    signal, fs = librosa.load(file_path, 16000)

    signal = np.array(signal, dtype=np.float64)

    # Get the duration of the audio file.
    duration = librosa.get_duration(signal, sr=fs)

    # Calculate the sample per track value
    sample_per_track = duration * fs
    # Calculate the number of sample segments with 5 seconds of segment duration
    number_of_samples_per_segment = segment_duration * fs
    # Calculate the number of possible segments for the file
    num_segments = int(sample_per_track // number_of_samples_per_segment)

    # Extract the features for each segment
    # set temp length in order to keep the safe file length
    # max_length = 0
    segments_feature = []
    segment_feature_name = []
    segments_f0 = []

    max_length = 0

    try:

        for s in range(num_segments):

            # Start the matlab engine
            eng = None #matlab.engine.start_matlab()

            # Start loading the toolboxes
            # Adding the related paths
            eng.addpath(MATLAB_ROOT, nargout=0)
            eng.addpath(VOICE_BOX, nargout=0)
            eng.addpath(VOICE_ANALYSIS_TOOLBOX, nargout=0)
            eng.addpath(DFA, nargout=0)
            eng.addpath(EMD, nargout=0)
            eng.addpath(RPDE, nargout=0)
            eng.addpath(SHRP, nargout=0)

            # Calculate the sample start point
            start_sample = number_of_samples_per_segment * s
            # Calculate the sample finish point
            finish_sample = start_sample + number_of_samples_per_segment

            sample = signal[start_sample:finish_sample]

            temp_segment_path = os.path.join(MEDIA_ROOT, "temp_segment.wav")

            sf.write(temp_segment_path, sample, fs)

            # Break the for loop if the last sample length is smaller than the rest.
            if max_length != 0:
                if len(sample) < max_length:
                    break
                else:
                    max_length = len(sample)
            else:
                max_length = len(sample)

            # Extract the features from matlab toolbox
            feature_vector_values, feature_vector_names, f0 = eng.voice_analysis(temp_segment_path,
                                                                                 nargout=3)

            feature_vector_values = np.array(feature_vector_values).flatten()
            f0 = np.array(f0).flatten()

            segment_feature_name = feature_vector_names
            segments_feature.append(feature_vector_values)
            segments_f0.append(f0)

            # Remove the file
            os.remove(temp_segment_path)

            # Store the extracted data in the target folder

            # Close the matlab engine
            eng.exit()

        gc.collect()

        return np.array(segment_feature_name), np.array(segments_feature), np.array(segments_f0)

    except Exception as ex:
        print(f"SystemError: {str(ex)}")
        return None, None, None


def praa_base_feature_downloader(file_path, segment_duration=2):
    """
    Download the mfcc features based on the information presented in data file

    the data file is going to keep the `file_name` and `user_id` and `gender`

    :param file_path Audio file path.
    :type file_path str
    :param segment_duration Segment duration in seconds
    :type segment_duration int
    :return: Return the tuples of three elements storing the features name and values based on the number of selected
    segments.
    :rtype: np.ndarray
    """

    try:


        signal, fs = librosa.load(file_path, 16000)

        signal = np.array(signal, dtype=np.float64)

        # Get the duration of the audio file.
        duration = librosa.get_duration(signal, sr=fs)

        # Calculate the sample per track value
        sample_per_track = duration * fs
        # Calculate the number of sample segments with 5 seconds of segment duration
        number_of_samples_per_segment = segment_duration * fs
        # Calculate the number of possible segments for the file
        num_segments = int(sample_per_track // number_of_samples_per_segment)

        # Extract the features for each segment
        # set temp length in order to keep the safe file length
        # max_length = 0
        segments_feature = []
        segment_feature_name = []
        segments_f0 = []

        max_length = 0

        for s in range(num_segments):

            # Calculate the sample start point
            start_sample = number_of_samples_per_segment * s
            # Calculate the sample finish point
            # Calculate the sample finish point
            finish_sample = start_sample + number_of_samples_per_segment

            sample = signal[start_sample:finish_sample]

            temp_segment_path = os.path.join(MEDIA_ROOT, "temp_segment_praat.wav")

            sf.write(temp_segment_path, sample, fs)

            # Break the for loop if the last sample length is smaller than the rest.
            if max_length != 0:
                if len(sample) < max_length:
                    break
                else:
                    max_length = len(sample)
            else:
                max_length = len(sample)

            # Extract the features from matlab toolbox
            feature_vector_values = extract_praa_based_features(temp_segment_path,
                                                                5,
                                                                1000,
                                                                "Hertz")

            segments_feature.append(feature_vector_values)

            # Remove the file
            os.remove(temp_segment_path)

        # Store the extracted data in the target folder
        gc.collect()

        return np.array(segments_feature)

    except Exception as ex:
        print(f"SystemError: {str(ex)}")
        return None


def extract_praa_based_features(voice_path, f0min, f0max, unit):
    """
    Extract the praa based features

    :param voice_path: Voice file path
    :type voice_path: str
    :param f0min: F0 minimum value
    :type f0min : int
    :param f0max: F0 maximum value
    :type f0max : int
    :param unit: Unit
    :type unit: str

    :return: List of the features extracted for each file
    :rtype : list
    """

    sound = parselmouth.Sound(voice_path)  # read the sound

    signal, fs = librosa.load(voice_path, 16000)

    # imf = emd.sift.sift(signal)

    # TODO add other features.

    mfcc_f = speechpy.feature.mfcc(signal, fs, num_filters=12)
    mfcc_f_mean = np.mean(mfcc_f.T, axis=1).tolist()
    mfcc_f_std = np.std(mfcc_f.T, axis=1).tolist()
    mfcc_f_d = speechpy.feature.extract_derivative_feature(mfcc_f)

    mfcc_f_mean_delta = np.mean(mfcc_f_d[:, :, 1].T, axis=1).tolist()
    mfcc_f_std_delta = np.std(mfcc_f_d[:, :, 1].T, axis=1).tolist()
    mfcc_f_mean_delta_delta = np.mean(mfcc_f_d[:, :, 2].T, axis=1).tolist()
    mfcc_f_std_delta_delta = np.std(mfcc_f_d[:, :, 2].T, axis=1).tolist()

    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)
    # pitch_cc = call(sound, "To Pitch (cc)", 0.0, f0min, f0max, "off", 0.3, 0.4, 0.01, 0.35, 0.14, 600)
    point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)  # create a praat pitch object
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    num_points = call(point_process, "Get number of points")
    matrix = call(sound, "To Harmonicity (gne)",
                  500., 4500.,
                  1000., 80.)

    final_feature_vector = mfcc_f_mean + mfcc_f_std
    final_feature_vector = final_feature_vector + mfcc_f_mean_delta + mfcc_f_std_delta
    final_feature_vector = final_feature_vector + mfcc_f_mean_delta_delta + mfcc_f_std_delta_delta

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []

    # Measure formants only at glottal pulses
    for point in range(0, num_points):
        point += 1
        t = call(point_process, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, unit, 'Linear')
        f2 = call(formants, "Get value at time", 2, t, unit, 'Linear')
        f3 = call(formants, "Get value at time", 3, t, unit, 'Linear')
        f4 = call(formants, "Get value at time", 4, t, unit, 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)

    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']

    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list)
    f2_mean = statistics.mean(f2_list)
    f3_mean = statistics.mean(f3_list)
    f4_mean = statistics.mean(f4_list)

    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    f1_median = statistics.median(f1_list)
    f2_median = statistics.median(f2_list)
    f3_median = statistics.median(f3_list)
    f4_median = statistics.median(f4_list)

    f0_voiced_fraction = call(pitch, 'Count voiced frames') / len(pitch)
    f0_q1_pitch = call(pitch, 'Get quantile',
                       0, 0,
                       0.25,
                       unit)

    f0_median_intensity = call(pitch, 'Get quantile',
                               0, 0,
                               0.50,
                               unit)
    f0_q3_pitch = call(pitch, 'Get quantile',
                       0, 0,
                       0.75,
                       unit)

    f0_mean_absolute_pitch_slope = call(pitch, 'Get mean absolute slope', unit)
    f0_pitch_slope_without_octave_jumps = call(pitch, 'Get slope without octave jumps')

    min_f0 = call(pitch, "Get minimum", 0, 0, unit, "Parabolic")  # get mean pitch
    f0_relative_min_pitch_time = call(pitch, 'Get time of minimum',
                                      0, 0,
                                      unit,
                                      "Parabolic") / sound.duration
    max_f0 = call(pitch, "Get maximum", 0, 0, unit, "Parabolic")  # get mean pitch
    f0_relative_max_pitch_time = call(pitch, 'Get time of maximum',
                                      0, 0,
                                      unit,
                                      "Parabolic") / sound.duration
    mean_f0 = call(pitch, "Get mean", 0, 0, unit)  # get mean pitch
    stdev_f0 = call(pitch, "Get standard deviation", 0, 0, unit)  # get standard deviation

    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    loca_labsolute_Jitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

    local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdb_shimmer = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    harmonicity05 = call(sound, "To Harmonicity (cc)", 0.01, 500, 0.1, 1.0)
    hnr05_min = call(harmonicity05, "Get minimum", 0, 0, "Parabolic")
    hnr05_min_time = call(harmonicity05, "Get time of minimum", 0, 0, "Parabolic") / sound.duration
    hnr05_max = call(harmonicity05, "Get maximum", 0, 0, "Parabolic")
    hnr05_max_time = call(harmonicity05, "Get time of maximum", 0, 0, "Parabolic") / sound.duration
    hnr05_mean = call(harmonicity05, "Get mean", 0, 0)
    hnr05_std = call(harmonicity05, "Get standard deviation", 0, 0)
    harmonicity15 = call(sound, "To Harmonicity (cc)", 0.01, 1500, 0.1, 1.0)
    hnr15_min = call(harmonicity15, "Get minimum", 0, 0, "Parabolic")
    hnr15_min_time = call(harmonicity15, "Get time of minimum", 0, 0, "Parabolic") / sound.duration
    hnr15_max = call(harmonicity15, "Get maximum", 0, 0, "Parabolic")
    hnr15_max_time = call(harmonicity15, "Get time of maximum", 0, 0, "Parabolic") / sound.duration
    hnr15_mean = call(harmonicity15, "Get mean", 0, 0)
    hnr15_std = call(harmonicity15, "Get standard deviation", 0, 0)
    harmonicity25 = call(sound, "To Harmonicity (cc)", 0.01, 2500, 0.1, 1.0)
    hnr25_min = call(harmonicity25, "Get minimum", 0, 0, "Parabolic")
    hnr25_min_time = call(harmonicity25, "Get time of minimum", 0, 0, "Parabolic") / sound.duration
    hnr25_max = call(harmonicity25, "Get maximum", 0, 0, "Parabolic")
    hnr25_max_time = call(harmonicity25, "Get time of maximum", 0, 0, "Parabolic") / sound.duration
    hnr25_mean = call(harmonicity25, "Get mean", 0, 0)
    hnr25_std = call(harmonicity25, "Get standard deviation", 0, 0)

    # harmonicity35 = call(sound, "To Harmonicity (cc)", 0.01, 3500, 0.1, 1.0)
    # hnr35 = call(harmonicity35, "Get mean", 0, 0)
    # harmonicity38 = call(sound, "To Harmonicity (cc)", 0.01, 3800, 0.1, 1.0)
    # hnr38 = call(harmonicity38, "Get mean", 0, 0)

    min_gne = call(matrix, 'Get minimum')
    max_gne = call(matrix, 'Get maximum')
    mean_gne = call(matrix, 'Get mean...',
                    0, 0,
                    0, 0)

    stddev_gne = call(matrix, 'Get standard deviation...',
                      0, 0,
                      0, 0)
    sum_gne = call(matrix, 'Get sum')

    final_feature_vector.append(np.float64(local_jitter))
    final_feature_vector.append(np.float64(loca_labsolute_Jitter))
    final_feature_vector.append(np.float64(rap_jitter))
    final_feature_vector.append(np.float64(ppq5_jitter))
    final_feature_vector.append(np.float64(local_shimmer))
    final_feature_vector.append(np.float64(localdb_shimmer))
    final_feature_vector.append(np.float64(apq3_shimmer))
    final_feature_vector.append(np.float64(aqpq5_shimmer))
    final_feature_vector.append(np.float64(apq11_shimmer))
    final_feature_vector.append(np.float64(hnr05_min))
    final_feature_vector.append(np.float64(hnr05_min_time))
    final_feature_vector.append(np.float64(hnr05_max))
    final_feature_vector.append(np.float64(hnr05_max_time))
    final_feature_vector.append(np.float64(hnr05_mean))
    final_feature_vector.append(np.float64(hnr05_std))
    final_feature_vector.append(np.float64(hnr15_min))
    final_feature_vector.append(np.float64(hnr15_min_time))
    final_feature_vector.append(np.float64(hnr15_max))
    final_feature_vector.append(np.float64(hnr15_max_time))
    final_feature_vector.append(np.float64(hnr15_mean))
    final_feature_vector.append(np.float64(hnr15_std))
    final_feature_vector.append(np.float64(hnr25_min))
    final_feature_vector.append(np.float64(hnr25_min_time))
    final_feature_vector.append(np.float64(hnr25_max))
    final_feature_vector.append(np.float64(hnr25_max_time))
    final_feature_vector.append(np.float64(hnr25_mean))
    final_feature_vector.append(np.float64(hnr25_std))
    # final_feature_vector.append(hnr35)
    # final_feature_vector.append(hnr38)

    final_feature_vector.append(np.float64(min_gne))
    final_feature_vector.append(np.float64(max_gne))
    final_feature_vector.append(np.float64(mean_gne))
    final_feature_vector.append(np.float64(stddev_gne))
    final_feature_vector.append(np.float64(sum_gne))

    final_feature_vector.append(np.float64(f0_voiced_fraction))
    final_feature_vector.append(np.float64(f0_q1_pitch))

    final_feature_vector.append(np.float64(f0_median_intensity))
    final_feature_vector.append(np.float64(f0_q3_pitch))

    final_feature_vector.append(np.float64(f0_mean_absolute_pitch_slope))
    final_feature_vector.append(np.float64(f0_pitch_slope_without_octave_jumps))

    final_feature_vector.append(np.float64(min_f0))
    final_feature_vector.append(np.float64(f0_relative_min_pitch_time))
    final_feature_vector.append(np.float64(max_f0))
    final_feature_vector.append(np.float64(f0_relative_max_pitch_time))
    final_feature_vector.append(np.float64(mean_f0))
    final_feature_vector.append(np.float64(stdev_f0))

    final_feature_vector.append(np.float64(f1_mean))
    final_feature_vector.append(np.float64(f2_mean))
    final_feature_vector.append(np.float64(f3_mean))
    final_feature_vector.append(np.float64(f4_mean))
    final_feature_vector.append(np.float64(f1_median))
    final_feature_vector.append(np.float64(f2_median))
    final_feature_vector.append(np.float64(f3_median))
    final_feature_vector.append(np.float64(f4_median))

    # final_feature_vector.append(np.int(imf.shape[1]))

    final_feature_vector = np.nan_to_num(final_feature_vector, nan=0)

    return final_feature_vector
