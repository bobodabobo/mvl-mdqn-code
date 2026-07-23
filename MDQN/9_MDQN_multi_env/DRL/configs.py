DQN_config = {
    "seed": 0,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "train_steps": 90000,
    "batch_size": 512,
    "len_epi_train": 50,
    "update_frq": 4,
    "lr": 0.0001,
    "cache_size": 5,
    "len_epi_eval": 200,
    "eval_times": 50,
    "n_repeats_eval": 1,
    "test_repeats": 100,
}


PPO_config = {
    **DQN_config,
    "rollout_size": DQN_config["batch_size"],
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "update_epochs": 4,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5
}

MDQN_config = {
    "H": 7,
    "w": 0.5,
    "train_steps": 50000,
    "eval_times": 50,
    "train_repeats": 2,
    "lambda_anc": 10.0 / 7.0,
    "eval_radius": 1
}
