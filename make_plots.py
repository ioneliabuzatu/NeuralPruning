import pandas as pd
import matplotlib.pyplot as plt


def lets_plot(file_analysis):
    """
    Plots will show which threshold is better for each layer based on the higher pick in the mAP accuracy
    :param file_analysis: the accuracies from run_analysis_layerwise with a data point as precison, recall, mAP and F1
    :return: plots for each layer
    """

    # x-axis
    pruning_percentiles_analysis = (5, 10, 20, 30, 40, 50, 60, 70, 80, 88)

    data = pd.read_pickle(file_analysis)
    for index, row in data.iterrows():
        precisions = [float(prec.split('|')[0]) for prec in row]
        recalls = [float(recall.split('|')[1]) for recall in row]
        mAPs = [float(map_.split('|')[2]) for map_ in row]
        f1s = [float(f1_.split('|')[3]) for f1_ in row]

        fig, ax = plt.subplots(2,2)
        fig.suptitle(f"conv.{index}")

        ax[0,0].plot(pruning_percentiles_analysis,mAPs, 'o-', color='red')
        ax[0, 0].set_xticks(pruning_percentiles_analysis)
        ax[0,0].set_title('mAP')

        ax[0, 1].plot(pruning_percentiles_analysis,recalls, '<-', color='green')
        ax[0, 1].set_xticks(pruning_percentiles_analysis)
        ax[0, 1].set_title('Recall')

        ax[1, 0].plot(pruning_percentiles_analysis,precisions, '>-', color='black')
        ax[1, 0].set_xticks(pruning_percentiles_analysis)
        ax[1, 0].set_title('Precision')

        ax[1, 1].plot(pruning_percentiles_analysis,f1s, '+-', color='yellow')
        ax[1, 1].set_xticks(pruning_percentiles_analysis)
        ax[1, 1].set_title('F1')

        plt.tight_layout()
        plt.savefig(f"./plots/conv_{index}")
        # plt.show()


lets_plot('./analysis.pkl')
