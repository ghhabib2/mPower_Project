import numpy as np
import statsmodels.api as sm
from scipy.stats import median_absolute_deviation, pearsonr
from statistics import mode as sys_mode
from librosa.feature import zero_crossing_rate
from astropy.timeseries import LombScargle


def lsp(x, t=None):
    """
    Calculate the Lomb-Scargle periodogram (LSP) of the input signal based on the sampling times.

    :param x: signal vector
    :param t: sampling times
    :return: Return the LSP of the signal
    """
    return LombScargle(t, x).autopower()


def zcr(x):
    return zero_crossing_rate(x)


def mode(x):
    """
    Calculate the the most frequent value of a list
    :param x: List to be evaluated
    :type x: list
    :return: return the mod value
    """
    return sys_mode(x)


def cor(x, y):
    return pearsonr(x, y)


def mad(x):
    return median_absolute_deviation(x)


def iqr_median_range_calculator(x):
    """
    Simultaneously calculating the iqr, median and range of a vector

    :param x: Target vector
    :type x: list
    :return: Return the tuple with the following format (iqr, median, range)
    :rtype: tuple
    """

    aux_quantile = np.quantile(x, q=[0, 0.25, 0.5, 0.75, 1])

    x_iqr = aux_quantile[3] - aux_quantile[1]
    x_median = aux_quantile[2]
    x_range = aux_quantile[4] - aux_quantile[0]

    return x_iqr, x_median, x_range


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
        y = np.power(x, 2) - np.concatenate((x[1:], [np.nan])) * np.concatenate(([np.nan], x[0:len(x) - 1]))
        return np.mean(y[np.logical_not(np.isnan(y))])


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


def acf(x, n_lags=2):
    if len(x) < 3:
        return None, None, None
    else:
        return sm.tsa.acf(x, nlags=n_lags)


def drift(x: list, y: list):
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))
