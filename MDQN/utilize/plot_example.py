import numpy as np
import pickle
import matplotlib.pyplot as plt

from .keys import systems, tasks, DQN_methods_short, heuristic_methods, colors, colors_heuristic
from DRL.configs import DQN_config

plt.rcParams['font.size'] = 7

def createExample():
    datas_DQN, datas_heuristic = _read_data()
    _plot_example(datas_DQN, datas_heuristic)

def _read_data(idx_system:int=0, idx_task:int=0):
    inv_sys = systems[idx_system]
    task = tasks[idx_task]
    # read DQN results
    DQN_file_name = "results/DQN_results.pkl"
    with open(DQN_file_name, "rb") as f:
        DQN_results = pickle.load(f)
    n_seeds = len(DQN_results[inv_sys][task][DQN_methods_short[0]])
    len_history = len(DQN_results[inv_sys][task][DQN_methods_short[0]][0]['history'])
    # init data matrix
    datas_DQN = np.zeros((len(DQN_methods_short), len_history), dtype=np.float64)
    # read DQN data
    for k, DQN_method in enumerate(DQN_methods_short):
        history_all_seeds = np.array([DQN_results[inv_sys][task][DQN_method][s]['history'] for s in range(n_seeds)])
        datas_DQN[k, :] = np.mean(history_all_seeds, axis=0)
    # read heuristic data
    heuristic_file_name = 'results/heuristic_results.pkl'
    with open(heuristic_file_name, "rb") as f:
        heuristic_results = pickle.load(f)
        datas_heuristic = []
        for heuristic_method in heuristic_methods[idx_system][1:]:
            performance = heuristic_results[inv_sys][task][heuristic_method]['performance']
            datas_heuristic.append((heuristic_method, performance))
        return datas_DQN, datas_heuristic
    
def _plot_example(datas_DQN, datas_heuristic):
    len_history = datas_DQN.shape[-1]
    frq = (DQN_config["train_steps"] + DQN_config['len_epi_eval'] * DQN_config['eval_times']) / (len_history - 1)
    x = np.arange(len_history) * frq
    fig, axs = plt.subplots(1, 1, sharey='row', figsize=(5, 2))
    for k, DQN_method in enumerate(DQN_methods_short):
        axs.plot(x, 
                 datas_DQN[k, :],
                 label=DQN_method, 
                 color=colors[k],
                 alpha=0.8,
                 linewidth=1)
    for i, (method, data) in enumerate(datas_heuristic):
        axs.axhline(y=data,
                    linestyle='--',
                    color=colors_heuristic[i],
                    alpha=0.8,
                    linewidth=1,
                    label=method)
    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("results/example.png", dpi=300)
    plt.savefig("results/example.pdf")