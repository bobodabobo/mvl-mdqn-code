# Code Release for MVL and MDQN

This directory contains the code prepared for the public release associated with the rebuttal.

## Structure

- `MVL/`: toy experiments for the MVL mechanism on Baird's counterexample.
- `MDQN/`: inventory-control experiments for the MDQN paper, including simulators, heuristic baselines, and DQN/MDQN implementations.

## Environment

The MDQN code was developed with Python and depends on:

- `numpy`
- `numba`
- `torch`
- `gymnasium`
- `openpyxl`
- `joblib`
- `matplotlib`

Install the MDQN dependencies with:

```bash
cd MDQN
pip install -r requirements.txt
```

## Running the main experiments

For the inventory experiments:

```bash
cd MDQN
python experiment_heuristic.py
python experiment_DQN.py
python summary.py
```

`autorun_all.sh` provides a convenience wrapper for the heuristic and DRL runs.

For the MVL toy example:

```bash
cd MVL
python main.py
```

## Notes

- The public release is intended to expose the runnable source code used in the study.
- Cache files, logs, and generated result artifacts should not be treated as source files for reproduction.
