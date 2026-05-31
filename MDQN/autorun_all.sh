nohup bash -c 'python -u experiment_heuristic.py > heuristic.log 2>&1 && python -u experiment_DQN.py > DQN.log 2>&1' > nohup.log 2>&1 &
