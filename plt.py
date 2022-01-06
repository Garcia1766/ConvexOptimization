import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def log(base, x):
    return np.log(x) / np.log(base)

def main():
    # FILENAME = f'admm_GRFrough_2333_5e-6_{sys.argv[1]}_{sys.argv[2]}.csv'
    # FILENAME = f'admm_test_data_2333_{sys.argv[1]}.csv'
    FILENAME = f'ipot_testd_data.csv'
    PLOT_NUM = 2000
    # GROUND_TRUTH = float(sys.argv[3])
    GROUND_TRUTH = 4
    all_df = pd.read_csv(FILENAME)
    print(all_df.values.shape)

    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['savefig.dpi'] = 2000
    plt.rcParams['figure.dpi'] = 150

    x = all_df.values[:PLOT_NUM, 0]
    fig, ax1 = plt.subplots()

    y1 = all_df.values[:PLOT_NUM, 1]
    y1 = log(10, y1)
    ax1.plot(x,y1,label='err',color='r', linewidth=0.5)
    ax1.set_xlabel("iter")
    ax1.set_ylabel('log(err)')

    ax2 = ax1.twinx()
    y2 = all_df.values[:PLOT_NUM, 2]
    y2 = log(10, abs(GROUND_TRUTH - y2))
    ax2.plot(x,y2,label='ans',color='b', linewidth=0.5)
    ax2.set_ylabel('log(4-ans)')

    plt.title(FILENAME[:-4])
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig('figs/' + FILENAME[:-4] + '.png')
    # plt.show()

if __name__ == '__main__':
    main()
