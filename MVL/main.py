import pickle
from algorithms import on_policy_TD_learning, on_policy_MVL, off_policy_MVL, off_policy_TD_learning
from plot import plot


def off_policy_learning(H=3, n_steps=1000, gamma=0.99, lr=0.01):
    log = dict()
    log['TD(FG)'] = off_policy_TD_learning(n_steps, gamma, lr, 'full')
    log['TD(SG)'] = off_policy_TD_learning(n_steps, gamma, lr, 'semi')
    log_mvl = off_policy_MVL(H, n_steps, lr)
    for h in range(H):
        log[f'MVL(h={h+1})'] = log_mvl[h]
    return log


def on_policy_learning(H=3, n_steps=1000, gamma=0.99, lr=0.01):
    log = dict()
    log['TD(FG)'] = on_policy_TD_learning(n_steps, gamma, lr, 'full')
    log['TD(SG)'] = on_policy_TD_learning(n_steps, gamma, lr, 'semi')
    log_mvl = on_policy_MVL(H, n_steps, lr)
    for h in range(H):
        log[f'MVL(h={h+1})'] = log_mvl[h]
    return log


if __name__ == '__main__':
    H = 3
    n_steps = 400
    gamma = 0.99
    lr = 0.01
    log = dict()
    log["on_policy"] = on_policy_learning(H, n_steps, gamma, lr)
    log["off_policy"] = off_policy_learning(H, n_steps, gamma, lr)
    pickle.dump(log, open("result.pkl", "wb"))
    with open("result.pkl", "rb") as f:
        log = pickle.load(f)
    plot(log)