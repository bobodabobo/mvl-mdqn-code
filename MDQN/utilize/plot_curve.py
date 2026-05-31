import numpy as np
import pickle
import matplotlib.pyplot as plt

from .keys import systems, tasks, DQN_methods_short, DQN_methods_long, colors
from DRL.configs import DQN_config

plt.rcParams['font.size'] = 7

def createCurve():
    datas = _read_data()
    _plot_all(datas)
    _plot_sub(datas)
    datas_long = _read_data_long()
    _plot_long(datas_long)

def _read_data():
    # read DQN results
    DQN_file_name = "results/DQN_results.pkl"
    with open(DQN_file_name, "rb") as f:
        DQN_results = pickle.load(f)
    n_seeds = len(DQN_results['LS']['1']['DQN'])
    len_history = len(DQN_results['LS']['1']['DQN'][0]['history'])
    # init data matrix
    datas = np.zeros((len(systems), len(tasks), len(DQN_methods_short), len_history), dtype=np.float64)
    # read data
    for i, inv_sys in enumerate(systems):
        for j, task in enumerate(tasks):
            for k, DQN_method in enumerate(DQN_methods_short):
                history_all_seeds = np.array([DQN_results[inv_sys][task][DQN_method][s]['history'] for s in range(n_seeds)])
                datas[i, j, k, :] = np.mean(history_all_seeds, axis=0)
    return datas

def _plot_all(data_mat:np.ndarray):
    len_history = data_mat.shape[-1]
    subfig_titles = [['(a)', '(b)', '(c)', '(d)'], ['(e)', '(f)', '(g)', '(h)'], ['(i)', '(j)', '(k)', '(l)']]
    frq = (DQN_config["train_steps"] + DQN_config['len_epi_eval'] * DQN_config['eval_times']) / (len_history - 1)
    x = np.arange(len_history) * frq
    fig, axs = plt.subplots(len(systems), len(tasks), sharey='row', figsize=(6.5, 5))
    for i, inv_sys in enumerate(systems):
        for j, task in enumerate(tasks):
            for k, DQN_method in enumerate(DQN_methods_short):
                axs[i, j].plot(x, 
                               data_mat[i, j, k, :],
                               label=DQN_method, 
                               color=colors[k],
                               alpha=0.8,
                               linewidth=1)
            axs[i, j].set_title(subfig_titles[i][j])
            # axs[i, j].grid(True)
            if not i == len(systems) - 1:
                axs[i, j].set_xticks([])
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    plt.tight_layout(rect=[0, 0.07, 1, 1]) 
    plt.savefig("results/curve.png", dpi=300)
    plt.savefig("results/curve.pdf")


def _plot_sub(data_mat:np.ndarray):
    len_history = data_mat.shape[-1]
    subfig_titles = ['(a)', '(b)', '(c)']
    frq = (DQN_config["train_steps"] + DQN_config['len_epi_eval'] * DQN_config['eval_times']) / (len_history - 1)
    x = np.arange(len_history) * frq
    fig, axs = plt.subplots(1, len(systems), figsize=(5.4, 1.8))
    for i, inv_sys in enumerate(systems):
        j = 0
        for k, DQN_method in enumerate(DQN_methods_short):
            axs[i].plot(x, 
                            data_mat[i, j, k, :],
                            label=DQN_method, 
                            color=colors[k],
                            alpha=0.8,
                            linewidth=1)
        axs[i].set_title(subfig_titles[i])
        axs[i].set_xticks([x[0], x[-1]])
        # axs[i].grid(True)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    plt.tight_layout(rect=[0, 0.1, 1, 1]) 
    plt.subplots_adjust(wspace=0.2) 
    plt.savefig("results/curve_sub.png", dpi=300)
    plt.savefig("results/curve_sub.pdf")

def _read_data_long():
    # read DQN results
    DQN_file_name = "results/DQN_results.pkl"
    with open(DQN_file_name, "rb") as f:
        DQN_results = pickle.load(f)
    n_seeds = len(DQN_results['LS']['1']['DQN-L'])
    len_history = len(DQN_results['LS']['1']['DQN-L'][0]['history'])
    # init data matrix
    datas = np.zeros((len(systems), len(tasks), len(DQN_methods_long), len_history), dtype=np.float64)
    # read data
    for i, inv_sys in enumerate(systems):
        for j, task in enumerate(tasks):
            for k, DQN_method in enumerate(DQN_methods_long):
                history_all_seeds = np.array([DQN_results[inv_sys][task][DQN_method][s]['history'] for s in range(n_seeds)])
                datas[i, j, k, :] = np.mean(history_all_seeds, axis=0)
    return datas

def _plot_long(data_mat:np.ndarray):
    len_history = data_mat.shape[-1]
    subfig_titles = [['(a)', '(b)', '(c)', '(d)'], ['(e)', '(f)', '(g)', '(h)'], ['(i)', '(j)', '(k)', '(l)']]
    frq = (DQN_config["train_steps_long"] + DQN_config['len_epi_eval'] * DQN_config['eval_times']) / (len_history - 1)
    x = np.arange(len_history) * frq
    fig, axs = plt.subplots(len(systems), len(tasks), sharey='row', figsize=(6.5, 5))
    for i, inv_sys in enumerate(systems):
        for j, task in enumerate(tasks):
            for k, DQN_method in enumerate(DQN_methods_long):
                axs[i, j].plot(x, 
                               data_mat[i, j, k, :],
                               label=DQN_method, 
                               color=colors[k],
                               alpha=0.8,
                               linewidth=1)
            axs[i, j].set_title(subfig_titles[i][j])
            if not i == len(systems) - 1:
                axs[i, j].set_xticks([])
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1]) 
    plt.savefig("results/curve_long.png", dpi=300)
    plt.savefig("results/curve_long.pdf")

if __name__ == "__main__":
    createCurve()