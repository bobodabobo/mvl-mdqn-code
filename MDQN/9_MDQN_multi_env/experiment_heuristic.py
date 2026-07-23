from pathlib import Path
import pickle

from heuristic import EchelonBaseStock
from heuristic import FixedOrderME
from heuristic import FixedOrderMI
from heuristic import SigmaSPolicy
from simulators import MultiItemInventory, multi_item_configs
from simulators import SerialMultiEchelonInventory, serial_multi_echelon_configs


RESULTS_PATH = Path("results/heuristic_results.pkl")
VALIDATION_PATH = Path("results/heuristic_validation_new_tasks.md")

SYSTEM_SPECS = [
    {
        "key": "ME",
        "title": "Multi-Echelon",
        "env_cls": SerialMultiEchelonInventory,
        "configs": serial_multi_echelon_configs,
        "policies": [
            ("EBS", "EchelonBaseStock", EchelonBaseStock),
            ("FO", "FixedOrderME", FixedOrderME),
        ],
    },
    {
        "key": "MI",
        "title": "Multi-Item",
        "env_cls": MultiItemInventory,
        "configs": multi_item_configs,
        "policies": [
            ("SigmaS", "SigmaSPolicy", SigmaSPolicy),
            ("FO", "FixedOrderMI", FixedOrderMI),
        ],
    },
]

VALIDATION_SPECS = {
    "ME": ("EBS", "FO", "EchelonBaseStock"),
    "MI": ("SigmaS", "FO", "SigmaSPolicy"),
}


def _format_parameters(parameters: dict):
    parts = []
    for key, value in parameters.items():
        if key == "sigma_states":
            preview = value[:5]
            parts.append(f"sigma_states={preview}... ({len(value)} states)")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _write_new_task_validation(results_dict: dict):
    lines = [
        "# Heuristic Validation for New Inventory Tasks",
        "",
        "| System | Task | Structural Policy | Structural Cost | Fixed Order Cost | Gap | Parameter Summary | Pass |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    all_pass = True
    for system_key, (structural_key, fixed_key, structural_name) in VALIDATION_SPECS.items():
        for task_id, task_results in results_dict[system_key].items():
            structural = task_results[structural_key]
            fixed = task_results[fixed_key]
            gap = fixed["performance"] - structural["performance"]
            is_pass = structural["performance"] < fixed["performance"]
            all_pass = all_pass and is_pass
            parameter_summary = _format_parameters(structural["parameters"])
            lines.append(
                f"| {system_key} | {task_id} | {structural_name} | "
                f"{structural['performance']:.4f} | {fixed['performance']:.4f} | "
                f"{gap:.4f} | {parameter_summary} | {'PASS' if is_pass else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            f"Overall validation: **{'PASS' if all_pass else 'FAIL'}**",
            "",
            "- Validation criterion: structural-policy cost must be strictly lower than fixed-order cost on every new task.",
        ]
    )
    VALIDATION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def heuristic_experiment(train_length: int = 1000, repeats: int = 100):
    print("===Heuristic Experiments===")
    results_dict = {}
    for system_spec in SYSTEM_SPECS:
        system_key = system_spec["key"]
        print(f"---{system_spec['title']}---")
        results_dict[system_key] = {}
        for task_id, config in enumerate(system_spec["configs"], start=1):
            print(f"-task-{task_id}-")
            results_dict[system_key][str(task_id)] = {}
            env = system_spec["env_cls"](config)
            for policy_key, policy_name, policy_cls in system_spec["policies"]:
                print(policy_name)
                agent = policy_cls(env)
                log = agent.train(length=train_length, repeats=repeats)
                results_dict[system_key][str(task_id)][policy_key] = log
                print(log)
        print()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("wb") as file:
        pickle.dump(results_dict, file)
    _write_new_task_validation(results_dict)


if __name__ == "__main__":
    heuristic_experiment()
