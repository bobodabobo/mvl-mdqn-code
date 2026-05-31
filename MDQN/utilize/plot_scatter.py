import numpy as np
import pickle
import matplotlib.pyplot as plt

from .keys import systems, tasks, DQN_methods, colors, markers

plt.rcParams['font.size'] = 7

def createScatter():
    datas = _read_data()
    _plot_all(datas)
    _plot_sub(datas)

def _read_data():
    # read DQN results
    DQN_file_name = "results/DQN_results.pkl"
    with open(DQN_file_name, "rb") as f:
        DQN_results = pickle.load(f)
    n_seeds = len(DQN_results['LS']['1']['DQN'])
    # init data matrix
    datas = np.zeros((len(systems), len(tasks), len(DQN_methods), n_seeds), dtype=np.float64)
    # read data
    for i, inv_sys in enumerate(systems):
        for j, task in enumerate(tasks):
            for k, DQN_method in enumerate(DQN_methods):
                for s in range(n_seeds):
                    datas[i, j, k, s] = DQN_results[inv_sys][task][DQN_method][s]['performance']
    return datas

def _plot_all(data_mat:np.ndarray):
    n_seeds = data_mat.shape[-1]
    subfig_titles = [['(a)', '(b)', '(c)', '(d)'], ['(e)', '(f)', '(g)', '(h)'], ['(i)', '(j)', '(k)', '(l)']]
    fig, axs = plt.subplots(len(systems), len(tasks), sharey='row', figsize=(5, 6))
    for i, inv_sys in enumerate(systems):
        for j, task in enumerate(tasks):
            for k, DQN_method in enumerate(DQN_methods):
                axs[i, j].scatter(
                    np.ones(n_seeds) * k, 
                    data_mat[i, j, k, :], 
                    label=DQN_method,
                    color=colors[k // 2],
                    marker=markers[k % len(markers)],
                    s=5
                )
            axs[i, j].set_title(subfig_titles[i][j])
            axs[i, j].set_xticks([])
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels) // 2 + 1, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 1]) 
    plt.savefig("results/scatter.png", dpi=300)
    plt.savefig("results/scatter.pdf")


def _plot_sub(data_mat:np.ndarray):
    n_seeds = data_mat.shape[-1]
    subfig_titles = ['(a)', '(b)', '(c)']
    fig, axs = plt.subplots(1, len(systems), figsize=(5, 2))
    for i, inv_sys in enumerate(systems):
        j = 0
        for k, DQN_method in enumerate(DQN_methods):
            axs[i].scatter(
                np.ones(n_seeds) * k, 
                data_mat[i, j, k, :], 
                label=DQN_method,
                color=colors[k // 2],
                marker=markers[k % len(markers)],
                s=5
            )
        axs[i].set_title(subfig_titles[i])
        axs[i].set_xticks([])
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels) // 2 + 1, frameon=False)
    plt.tight_layout(rect=[0, 0.15, 1, 1]) 
    plt.savefig("results/scatter_sub.png", dpi=300)
    plt.savefig("results/scatter_sub.pdf")


if __name__ == "__main__":
    createScatter()