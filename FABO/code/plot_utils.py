import numpy as np 
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torch
import seaborn as sns
from sklearn.decomposition import PCA
import os
from utils import data_load

plt.matplotlib.style.use("https://gist.githubusercontent.com/JonnyCBB/c464d302fefce4722fe6cf5f461114ea/raw/64a78942d3f7b4b5054902f2cee84213eaff872f/matplotlibrc")

def get_lines_for_plot(res, X, y, rank=True):
    
    # Getting rankings of labels
    y = np.array(y)
    ids_to_rank = np.argsort(y.squeeze())
    y_ranks = np.arange(len(X.index))[np.flip(ids_to_rank).argsort()] + 1


    y_max_mu5b      = np.zeros(res["ids_acquired"].shape[0])
    y_max_sig_bot5b = np.zeros(res["ids_acquired"].shape[0])
    y_max_sig_top5b = np.zeros(res["ids_acquired"].shape[0])
    for i in range(1, res["ids_acquired"].shape[0]+1):
        # max value acquired up to this point
        if rank:
            y_maxes5b = np.array([np.min(y_ranks[res['ids_acquired'][:,r].astype("int64")][:i]) for r in range(res["ids_acquired"].shape[1])]) # among runs
        else:
            y_maxes5b = np.array([np.max(y[res['ids_acquired'][r]][:i]) for r in range(res["ids_acquired"].shape[1])]) # among runs
        assert np.size(y_maxes5b) == res["ids_acquired"].shape[1]
        y_max_mu5b[i-1]      = np.mean(y_maxes5b)
        y_max_sig_bot5b[i-1] = np.std(y_maxes5b[y_maxes5b < y_max_mu5b[i-1]])
        y_max_sig_top5b[i-1] = np.std(y_maxes5b[y_maxes5b > y_max_mu5b[i-1]])
    line = y_max_mu5b
    lower_uncertainity = y_max_mu5b - y_max_sig_bot5b
    upper_uncertainity = y_max_mu5b + y_max_sig_top5b
    return line, lower_uncertainity, upper_uncertainity


def plot_experiment(res, line, lower_uncertainity, upper_uncertainity, dataset_name, data, axs, legend, hist=False):
    axs[0].plot(range(res["ids_acquired"].shape[0]), line, label=legend, clip_on=False)
    axs[0].fill_between(range(res["ids_acquired"].shape[0]), lower_uncertainity, 
                                            (upper_uncertainity), 
                        alpha=0.2, ec="None")


    axs[0].set_xlabel('# evaluated MOFs')
    axs[0].set_ylabel('Minimum Band Gap\namong acquired MOFs\n')


    axs[0].set_ylabel('highest rank\namong acquired MOFs')
    axs[0].set_title('Best MOF Rank vs. Number of Iterations',fontsize=12)
    axs[0].set_xlim([0, range(res["ids_acquired"].shape[0])[-1]+1])
    axs[0].legend(fontsize="7")
    axs[0].axhline(y=1, color="k", linestyle="--", zorder=0) # to see the band bleed into negative zone.
    axs[0].set_yscale("log")
    ylim_ax0 = axs[0].get_ylim()

    if hist:
        data = data*-1
        if dataset_name == "CO2 Uptake" or dataset_name == "COFS":
            data = data*-1
        hist, bins = np.histogram(data, bins=32)
        axs[1].hist(data, orientation="horizontal",bins=bins,color='#00BEFF',alpha=0.5, edgecolor = "black")
        axs[1].set_xlabel("Count")
        if dataset_name == "CO2 Uptake" or dataset_name == "COFS":
            axs[1].set_ylabel("CO2 Uptake (mol/kg)")
        else:
            axs[1].set_ylabel("MOF Band Gap")
        axs[1].set_title('{} Dataset'.format(dataset_name),fontsize=12)
        xlim_ax1 = axs[1].get_xlim()

    plt.tight_layout()
    plt.savefig("{}.pdf".format(dataset_name))

def generate_rank_plot(to_plot, dataset_name, legend, iterations=250):
    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=[1.2 * 6.4, 4.8])#, sharey=True)
    X, y = data_load.load_dataset(dataset_name)

    for i in range(len(to_plot)):

        with open(to_plot[i], 'rb') as f:
            res = pickle.load(f)

        line, lower_uncertainity, upper_uncertainity = get_lines_for_plot(res, X, y)
        data = y
        if i + 1 == len(to_plot):
            hist = True
        else:
            hist = False
        # print()
        plot_experiment(res, line, lower_uncertainity, upper_uncertainity, dataset_name, data, axs, legend[i], hist=hist)

def process_pure_discovery_data(exp,base_dir="ranking_for_plot"):
    new_path = base_dir + "/Pure_Discovery/" + exp + ".pkl"


    ranks = []
    feature_count = []
    ids = []

    for i in os.listdir(base_dir):
        if exp in i:
            for j in os.listdir(base_dir + "/" + i):
                path = base_dir + "/" + i + "/" + j
                
                with open(path, 'rb') as f:
                    data = pickle.load(f)

                mydict = data[4]

                ranks.append(mydict["Rank"])
                feature_count.append(mydict["Feature Count"])
                ids.append(mydict["IDS"])

    rank = np.zeros(np.expand_dims(ranks[0],1).shape)
    feature_count_array = np.zeros(np.expand_dims(feature_count[0],1).shape)
    ids_array = np.zeros(np.expand_dims(ids[0],1).shape)

    for r in range(len(ranks)):
        rank = np.concatenate((rank,np.expand_dims(ranks[r],axis=1)),axis=1)
        ids_array = np.concatenate((ids_array,np.expand_dims(ids[r],axis=1)),axis=1)
        feature_count_array = np.concatenate((feature_count_array,np.expand_dims(feature_count[r],axis=1)),axis=1)

    rank = rank[:,1:]
    ids_array = ids_array[:,1:]
    feature_count_array = feature_count_array[:,1:]

    ids_array.astype("int64")


    save_dict = {"rank" : rank, "ids_acquired" : ids_array, "feature count" : feature_count_array}

    if not os.path.exists(base_dir + "/Pure_Discovery/"):
        os.mkdir(base_dir + "/Pure_Discovery/")

    file = open(new_path, 'wb')
    # dump information to that file
    pickle.dump(save_dict,file)
    # close the file
    file.close()

def get_top_one_percent(to_plot, dataset_name):
    X, y = data_load.load_dataset("PBE")

    top_percent = []

    for i in range(len(to_plot)):
        # print(to_plot[i])

        with open(to_plot[i], 'rb') as f:
            res = pickle.load(f)

        MOF_count = len(X.index)
        if "Mean" in res.keys():
            top_percent.append(sum(res["Mean"] < MOF_count/100)/len(res["Mean"]))
        else:
            mean = np.mean(res["rank"],axis=1)
            top_percent.append(sum(mean < MOF_count/100)/len(mean))

    return top_percent