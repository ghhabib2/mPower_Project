# Define the libraries Roots
import os
import numpy as np
import matlab.engine
import soundfile as sf
import librosa
import time

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
            eng = matlab.engine.start_matlab()

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

            time.sleep(2)
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

        return np.array(segment_feature_name), np.array(segments_feature), np.array(segments_f0)

    except IOError as ex:
        print(f"IOError: {str(ex)}")
        return None, None, None
