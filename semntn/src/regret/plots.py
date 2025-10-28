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
    y = np.array(regrets, dtype=float)
    # linear fit: y ~ k * x + b
    k, b = np.polyfit(x, y, deg=1)
    yfit = k * x + b
    plt.figure()
    plt.plot(x, y, label="R(T)")
    plt.plot(x, yfit, linestyle="--", label=f"fit: k={k:.2f}")
    plt.xlabel("log T")
    plt.ylabel("Cumulative Regret R(T)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_regret_over_logT(regrets, path, dpi=150):
    t = np.arange(1, len(regrets) + 1)
    xlog = np.log(t)
    y = np.array(regrets, dtype=float) / np.maximum(xlog, 1e-9)
    plt.figure()
    plt.plot(t, y)
    plt.xlabel("T")
    plt.ylabel("R(T) / log T")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
