import pandas as pd
import matplotlib.pyplot as plt


def lets_plot(file_analysis):

    data = pd.read_pickle(file_analysis)
    precisions = []
    recalls = []
    mAPs = []
    f1s = []
    for layer in data.iterrows():

        print(layer)


        # precision =
        # recall =
        # mAP =
        # f1 =
        #
        # precisions.append(precision)
        # recalls.append(recall)
        # mAPs.append(mAP)
        # f1s.append(f1)
        #
        #
        # fig, ax = plt.subplots(2,2)
        #
        # ax[0,0].plot(mAPs)
        # ax[0,1].plot(recalls)
        # ax[1,0].plot(precisions)
        # ax[1,1].plot(f1s)
        #
        #
        # # plt.savefig(f"./plots/layer_name_{layer}")
        # plt.show()
