import matplotlib.pyplot as plt
import numpy as np

def signal_plotter(time, data, title:str):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    t = np.array(time)

    # generate noise:
    s = np.array(data)
    fig, axs = plt.subplots()
    axs.set_title(title)
    axs.plot(t, s, color='C0')
    axs.set_xlabel("Time")
    axs.set_ylabel("Amplitude")
    plt.show()
    plt.clf()

def array_plot(data, title:str):
    n = len(data)
    x = np.array(range(0, n))
    y = np.array(data)
    plt.title(title)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.plot(x, y, color='C0')
    plt.show()