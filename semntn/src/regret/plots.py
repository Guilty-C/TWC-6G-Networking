import numpy as np
import matplotlib.pyplot as plt


def plot_regret_T(regrets, path, dpi=150):
    t = np.arange(1, len(regrets) + 1)
    plt.figure()
    plt.plot(t, regrets)
    plt.xlabel("T")
    plt.ylabel("Cumulative Regret R(T)")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_regret_logT(regrets, path, dpi=150):
    x = np.log(np.arange(1, len(regrets) + 1))
    plt.figure()
    plt.plot(x, regrets)
    plt.xlabel("log T")
    plt.ylabel("Cumulative Regret R(T)")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
