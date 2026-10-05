"""Evaluate the trained paper configurations with fixed initial states."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .baselines import METHODS, run_baseline
from .io import create_output_directory, runtime_versions, sha256, summarize, write_csv, write_json
from .metrics import calculate_final_oprr, calculate_total_variance, strict_reversal_counts
from .protocol import (ABLATIONS, DISTRIBUTIONS, EPSILONS, EVALUATION_SEEDS,
                       OOD_EPSILONS, ORDINAL_WEIGHTS, SCALES, STABILITY_SEEDS,
                       TYPICAL_EPSILONS, make_env, ordinal_configuration,
                       scale_configuration)
from .sampling import sample_initial_opinions


SECTIONS = ("case", "baselines", "ordinal", "ablation", "sensitivity",
            "scalability", "stability", "epsilon_ood", "distribution_ood")


def episode_jobs(section: str) -> list[dict]:
    jobs = []
    def add(configuration, epsilons, seeds, n=100, m=5,
            distribution="uniform_control", rng="default_rng", **extra):
        for epsilon in epsilons:
            for seed in seeds:
                jobs.append(dict(configuration=configuration, epsilon=epsilon, seed=seed,
                                 N=n, M=m, distribution=distribution, rng=rng, **extra))
    if section == "case":
        add("main", TYPICAL_EPSILONS, (42,), rng="random_state")
    elif section == "baselines":
        add("main", TYPICAL_EPSILONS, EVALUATION_SEEDS)
    elif section == "ordinal":
        add("main", TYPICAL_EPSILONS, EVALUATION_SEEDS)
    elif section == "ablation":
        for configuration in ("main", *ABLATIONS):
            add(configuration, (0.05,), EVALUATION_SEEDS)
    elif section == "sensitivity":
        for weight in ORDINAL_WEIGHTS:
            add(ordinal_configuration(weight), TYPICAL_EPSILONS, EVALUATION_SEEDS,
                lambda_rev=weight)
    elif section == "scalability":
        for n, m in SCALES:
            add(scale_configuration(n, m), EPSILONS, (42,), n=n, m=m)
    elif section == "stability":
        add("main", (0.05,), STABILITY_SEEDS)
    elif section == "epsilon_ood":
        add("main", OOD_EPSILONS, EVALUATION_SEEDS)
    elif section == "distribution_ood":
        for distribution in DISTRIBUTIONS:
            add("main", EPSILONS, EVALUATION_SEEDS, distribution=distribution)
    else:
        raise ValueError(f"Unknown section: {section}")
    return jobs


def run_ppo_episode(model, configuration: str, initial: np.ndarray,
                    epsilon: float, seed: int, record: bool = False):
    env = make_env(configuration)
    if model.observation_space.shape != env.observation_space.shape:
        raise ValueError(f"Checkpoint observation shape does not match {configuration}")
    if model.action_space.shape != env.action_space.shape:
        raise ValueError(f"Checkpoint action shape does not match {configuration}")
    observation, _ = env.reset(seed=seed, options={
        "fixed_initial_state": initial, "fixed_epsilon": epsilon,
    })
    if not np.array_equal(env.U_0, initial):
        raise RuntimeError("Initial-state injection changed the input matrix")
    traces = [{"model": "PPO-HK", "epsilon": epsilon, "micro_step": 0,
               "equivalent_rounds": 0.0, "variance": env.total_variance,
               "gravity_shift": 0.0}] if record else []
    opinions = [(0.0, initial.copy())] if record else []
    steps, reward_sum = 0, 0.0
    terminated = truncated = False
    try:
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
            reward_sum += float(reward)
            if record:
                traces.append(dict(model="PPO-HK", epsilon=epsilon, micro_step=steps,
                                   equivalent_rounds=steps / env.N,
                                   variance=float(info["variance"]),
                                   gravity_shift=float(info["current_gravity"])))
                if info["hk_applied"] or terminated or truncated:
                    opinions.append((steps / env.N, env.U_t.copy()))
        final = env.U_t.copy()
        valid, reversals, ties = strict_reversal_counts(initial, final)
        row = dict(
            model="full_model" if configuration == "main" else configuration,
            N=env.N, M=env.M, epsilon=epsilon, seed=seed,
            initial_variance=calculate_total_variance(initial),
            intervention_steps=steps, equivalent_rounds=steps / env.N,
            completed_synchronous_updates=env.current_round,
            remaining_micro_steps=env.current_expert,
            final_variance=env.total_variance, gravity_shift=env.gravity_shift,
            final_oprr=calculate_final_oprr(initial, final),
            joint_success=bool(terminated), variance_feasible=bool(info["variance_feasible"]),
            gravity_feasible=bool(info["gravity_feasible"]),
            valid_pairs=int(valid.sum()), reversal_count=int(reversals.sum()),
            terminal_tied_pairs=int(ties.sum()), episode_reward=reward_sum,
            stop_reason="joint_success" if terminated else "max_rounds",
            failure_reason="" if terminated else (
                "variance_and_gravity" if not info["variance_feasible"] and not info["gravity_feasible"]
                else "variance" if not info["variance_feasible"] else "gravity"),
            lambda_rev=env.reward_config.ordinal_weight,
        )
        return row, traces, opinions, final
    finally:
        env.close()


def rank_text(vector: np.ndarray) -> str:
    order = np.argsort(-vector, kind="stable")
    text = str(order[0] + 1)
    for previous, current in zip(order[:-1], order[1:]):
        text += ("=" if vector[previous] == vector[current] else "≻") + str(current + 1)
    return text


def reversal_pairs(initial: np.ndarray, final: np.ndarray) -> list[tuple]:
    return [(a + 1, b + 1) for a in range(len(initial)) for b in range(len(initial))
            if initial[a] > initial[b] and final[a] < final[b]]


def expert_rows(initial: np.ndarray, hk: np.ndarray, ppo: np.ndarray) -> list[dict]:
    valid, hk_reversals, _ = strict_reversal_counts(initial, hk)
    _, ppo_reversals, _ = strict_reversal_counts(initial, ppo)
    rows = []
    for index in range(initial.shape[0]):
        row = dict(
            expert_id=f"专家{index + 1:02d}",
            initial_vector="(" + ", ".join(f"{x:.9f}" for x in initial[index]) + ")",
            initial_rank=rank_text(initial[index]),
            hk_final_vector="(" + ", ".join(f"{x:.9f}" for x in hk[index]) + ")",
            hk_final_rank=rank_text(hk[index]),
            ppo_final_vector="(" + ", ".join(f"{x:.9f}" for x in ppo[index]) + ")",
            ppo_final_rank=rank_text(ppo[index]), valid_pairs=int(valid[index]),
            hk_reversal_count=int(hk_reversals[index]),
            hk_oprr=100.0 * hk_reversals[index] / valid[index] if valid[index] else 0.0,
            ppo_reversal_count=int(ppo_reversals[index]),
            ppo_oprr=100.0 * ppo_reversals[index] / valid[index] if valid[index] else 0.0,
            hk_reversed_pairs=str(reversal_pairs(initial[index], hk[index])),
            ppo_reversed_pairs=str(reversal_pairs(initial[index], ppo[index])),
        )
        for prefix, matrix in (("initial", initial), ("hk_final", hk), ("ppo_final", ppo)):
            row.update({f"{prefix}_u{k + 1}": float(value) for k, value in enumerate(matrix[index])})
        rows.append(row)
    return rows


def opinion_rows(method: str, states: list[tuple]) -> list[dict]:
    rows = []
    for time, matrix in states:
        for expert, vector in enumerate(matrix, 1):
            rows.append(dict(method=method, time=float(time), expert=expert,
                             **{f"u{k + 1}": float(value) for k, value in enumerate(vector)}))
    return rows


def evaluate(section: str, model_dir: Path, output: Path,
             main_checkpoint: Path | None, natural_only: bool, device: str):
    if natural_only and section != "baselines":
        raise ValueError("--natural-only applies only to the baselines section")
    jobs = episode_jobs(section)
    checkpoint_paths = {}
    if not natural_only:
        for configuration in dict.fromkeys(job["configuration"] for job in jobs):
            path = (main_checkpoint if configuration == "main" and main_checkpoint is not None
                    else model_dir / configuration / "final_model.zip")
            if not path.is_file():
                raise FileNotFoundError(f"Train {configuration} first: missing {path}")
            checkpoint_paths[configuration] = path
    output = create_output_directory(output)
    torch.set_num_threads(1)
    models = {key: PPO.load(str(path), device=device) for key, path in checkpoint_paths.items()}
    write_json(output / "protocol.json", {
        "section": section, "deterministic": True, "runtime": runtime_versions(),
        "checkpoints": {key: {"filename": path.name, "sha256": sha256(path)}
                        for key, path in checkpoint_paths.items()},
        "jobs": jobs, "natural_only": natural_only,
        "terminal_condition": {"variance_threshold": 0.05, "gravity_threshold": 0.45},
        "sample_std_ddof": 1,
    })
    rows, variance_traces, trajectory_rows = [], [], []
    for count, job in enumerate(jobs, 1):
        initial = sample_initial_opinions(job["distribution"], job["seed"],
                                          job["N"], job["M"], job["rng"])
        configuration = job["configuration"]
        ppo_final = None
        if not natural_only:
            row, traces, states, ppo_final = run_ppo_episode(
                models[configuration], configuration, initial, job["epsilon"], job["seed"],
                record=section == "case")
            row.update(section=section, rng=job["rng"], distribution=job["distribution"])
            rows.append(row)
            variance_traces.extend(traces)
            if section == "case" and job["epsilon"] == 0.05:
                trajectory_rows.extend(opinion_rows("PPO-HK", states))
        if section in ("case", "baselines"):
            for method in (("hk",) if section == "case" else METHODS):
                row, trajectory = run_baseline(
                    method, initial, job["epsilon"], job["seed"],
                    record_trajectory=section == "case")
                row.update(section=section, model=method, intervention_steps=0,
                           rng=job["rng"], distribution=job["distribution"])
                if not row["joint_success"]:
                    row["failure_reason"] = (
                        "variance_and_gravity" if not row["variance_feasible"] and not row["gravity_feasible"]
                        else "variance" if not row["variance_feasible"] else "gravity")
                rows.append(row)
                if section == "case":
                    for state in trajectory:
                        variance_traces.append(dict(
                            model="HK", epsilon=job["epsilon"],
                            micro_step=state["round"] * job["N"],
                            equivalent_rounds=state["round"], variance=state["variance"],
                            gravity_shift=state["gravity"]))
                    if job["epsilon"] == 0.05:
                        states = [(state["round"], state["opinions"]) for state in trajectory]
                        trajectory_rows.extend(opinion_rows("HK", states))
                        write_csv(output / "expert_preferences.csv", expert_rows(
                            initial, trajectory[-1]["opinions"], ppo_final))
        print(f"{section}: {count}/{len(jobs)} initial-state evaluations", flush=True)
    write_csv(output / "episode_results.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    failures = [row for row in rows if not row["joint_success"]]
    if failures:
        write_csv(output / "failed_episodes.csv", failures)
    if variance_traces:
        write_csv(output / "variance_trajectories.csv", variance_traces)
        write_csv(output / "opinion_trajectories.csv", trajectory_rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=SECTIONS, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("runs"))
    parser.add_argument("--main-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--natural-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(evaluate(args.section, args.model_dir, args.output, args.main_checkpoint,
                   args.natural_only, args.device))


if __name__ == "__main__":
    main()
