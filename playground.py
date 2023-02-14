import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14,
    }
)


def main():
    x = np.array([1, 5, 10, 50, 100, 500, 1000])
    y = np.array([1.02, .35, .14, .05, .03, .023, .02])
    plt.plot(x, y)
    plt.yscale('log')
    plt.xlabel('epochs jitted')
    plt.ylabel('time per epoch (seconds)')
    plt.title('jitting epochs performance (small robust least squares)')
    plt.savefig('playground_img/jit_epochs.pdf')


if __name__ == "__main__":
    main()
