# Paper experiment map

This package contains the archived data displayed in Sections 3.2--3.8. The data
were mechanically exported without training, model inference, smoothing, or
changes to the original archives. New training runs need not reproduce exactly
the same learned policy or episode values. `paper_results/provenance.json`
records revision_code-relative source names, SHA256 checksums, and selection
rules; it contains no local absolute paths.

| Paper section | Published data | Coverage |
| --- | --- | --- |
| 3.2 Training behavior | `section_32_monitor.csv` | Complete main-model training episode monitor; original leading metadata comment and r/l/t columns retained. |
| 3.3.1 Variance comparison | `section_33_variance_trajectories.csv`, `section_33_performance_raw.csv` | Original 2877 variance rows and six endpoint rows, epsilon 0.05/0.15/0.25, RandomState(42) initial matrix. |
| 3.3.2 Opinion trajectories | `section_332_opinion_trajectories.csv` | HK: 201 saved states; full model: 14 saved states including the actual terminal partial sweep. 100 experts x 5 alternatives; 21500 rows. |
| 3.3.3 HK extensions | `section_333_baselines.csv` | Five natural HK methods plus full model, three thresholds x 20 seeds; 360 rows. |
| 3.4.1 OPRR statistics | Full-model rows in `section_333_baselines.csv` | Same 60 full-model evaluation episodes as the three-threshold comparison. |
| 3.4.2 Expert preferences | `section_342_experts.csv` | All 100 experts and their original/HK/full-model preference and reversal fields; no expert selection was applied. |
| 3.5 Mechanism ablation | `section_35_ablation.csv` | Full model, no_mask, no_gravity_penalty; epsilon 0.05 x 20 paired seeds; 60 rows. |
| 3.6 Ordinal-weight sensitivity | `section_36_sensitivity.csv` | lambda_rev 0.5/1/1.5/2 x three thresholds x 20 seeds; 240 rows. |
| 3.7 Scale experiments | `section_37_scalability.csv` | (50,5), (100,5), (200,5), (100,3), (100,10) x six thresholds; evaluation seed 42; 30 rows. |
| 3.8.1 Unseen ID initial states | `section_381_id_states.csv` | epsilon 0.05, seeds 10000--10099; 100 rows. |
| 3.8.2 Parameter OOD | `section_382_ood_epsilon.csv` | epsilon 0.03 and 0.35 x 20 seeds; 40 rows. |
| 3.8.3 Distribution OOD | `section_383_ood_distributions.csv` | Four paper distributions x six thresholds x 20 seeds; 480 rows. |

## Data fields and units

Evaluation CSVs share `section,model,N,M,epsilon,seed,rng,distribution,
intervention_steps,equivalent_rounds,final_variance,gravity_shift,final_oprr,
joint_success`. Additional archived diagnostics are retained when present.
Sensitivity results include `lambda_rev`.

`final_oprr` is a percentage, not a fraction. `final_variance` is the sum of
population variances across alternatives. `gravity_shift` is normalized by
sqrt(M). The success test is V <= 0.05 and G <= 0.45. Failed episodes remain
in the exports. Mean successful intervention counts use only success episodes;
terminal metric summaries use all selected episodes. Evaluation standard
deviations use ddof=1.

Natural HK methods have `intervention_steps=0`, while their equivalent rounds
are actual synchronous updates. PPO equivalent rounds equal individual
interventions divided by N. The typical case stops at 1271 interventions,
12 completed HK sweeps and 71 remaining interventions, i.e. 12.71 equivalent
rounds. These quantities are not computation time.

The opinion trajectory CSV uses `method,time,expert,u1,u2,u3,u4,u5`. Expert
indexes are one-based. HK time is its synchronous-update index; full-model time
is its stored equivalent-round coordinate. The 14 full-model states are the
saved snapshots, not a newly interpolated 1272-state trajectory. Post-terminal
passive HK continuation is excluded.

## Configuration and random generators

Configuration IDs are `main`, `no_mask`, `no_gravity_penalty`,
`lambda_rev_0p5`, `lambda_rev_1p5`, `lambda_rev_2`, `N_50_M_5`, `N_200_M_5`,
`N_100_M_3`, and `N_100_M_10`. Weight 1 and shape (100,5) reuse `main`.
The `main` CSV model label is `full_model`; other trained variants use their
configuration IDs. New checkpoint paths default to
`runs/{configuration}/final_model.zip`.

All archived training configurations record seed 42, a 10,000,000-step budget,
n_steps=2048, batch_size=1024 and one environment. The completed archived runs
contain 10,000,384 steps. `configs/paper_protocol.json` records the environment,
reward and PPO settings. Main data originate from the independently trained
gravity-terminal (100,5) archive.

The typical Sections 3.3/3.4.2 example uses RandomState(42). The evaluation
generators for the other experiments use default_rng(seed), including the
scale example with seed 42. Identical seed numbers do not make these two
generators produce identical initial opinions. Training resets draw iid
Uniform(0,1) opinions and an episode-fixed epsilon from Uniform(0.05,0.30).

The ablation `no_gravity_penalty` removes only the process reward coefficient;
it retains the terminal G <= 0.45 test. Scale results use policies trained for
their respective shapes and do not demonstrate one-policy zero-shot transfer
across shapes. ID initial-state tests keep the training distribution. Parameter
OOD tests change only epsilon to two specified values; distribution OOD tests
use Beta(2,2), Beta(0.5,0.5), Beta(5,2), and a row-permuted two-group
Beta(2,8)/Beta(8,2) construction. They describe the tested conditions.

## Original archive plotting entry points

These names document the source archive and data transformations. They are
not a requirement to copy platform-specific archive scripts into the release.

| Figure | Original archive scripts and dependencies |
| --- | --- |
| Training curves | `plot_section_32_training_dual_bottom.py` imports `plot_section_32_training_dual.py` and `plot_section_32_training.py`; 100000-step episode bins and three-bin episode-count-weighted smoothing. |
| Variance curves | `plot_section_331_variance_refined.py` reads archived variance_trajectories.csv; `plot_section_331_variance_comparison.py` was its extraction/evaluation source. |
| Opinion trajectories | `plot_section_332_final.py` reads the saved NPZ from `plot_section_332_all_alternatives.py`; the release CSV preserves only actual HK/full-model episode histories. |
| Weight sensitivity | `plot_ordinal_weight_sensitivity_updated.py` reads the archived 240-episode summary. |
| Scale curves | `plot_section_37_scalability_final.py` imports style definitions from `plot_section_37_scalability.py` and reads the archived 30-row scale table. |
| ID initial states | `plot_section_381_eps005_original_style.py` imports helpers from `plot_section_381_distribution.py`; display jitter has seed 381 and does not alter vertical data. |

No additional baselines, reward variants, initial distributions or exploratory
seed windows are included in these published evaluation tables.
