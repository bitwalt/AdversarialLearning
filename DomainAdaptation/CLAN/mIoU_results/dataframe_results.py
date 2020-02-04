import pandas as pd
import matplotlib.pyplot as plt

def main(file):

    steps = []
    mious = []
    with open(file) as f:
        for line in f.readlines():
            l = line.split("step_{%d}\t\t===> mIoU: %d")

    df = pd.DataFrame(data, columns=['First Column Name', 'Second Column Name'])
    print(df)


def plot_results(steps, mious):




if __name__ == '__main__':
    file = './Deeplab_10k_GTA'
    main()
