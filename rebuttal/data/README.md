# JointLoom rebuttal data and configurations

The revised paper's Appendix A contains all S tables and Figures S1/S2.
This package supplies reported values, per-seed/per-task records, protocols,
configuration selection and plot sources. It contains no implementation archive.
Code uses the original anonymous entry point listed in IMPLEMENTATION_INDEX.md.
The ZIP and its extracted files are mirrored under rebuttal/data/ at that link.

| Evidence | Revised PDF page | Data/configuration group |
| --- | --- | --- |
| S1 | 13 | data/reported-results.csv; magat-online-input-wait-*; magat-fixed-affinity-measurements.json; magat-random-ready-measurements.json; config/configuration-selection/ |
| S2 | 14 | data/gpu-input-reduction-* |
| S3 | 14 | data/magat-recorded-consumption-* |
| S4b | 15 | data/magat-original-checkpoints-* |
| S5 | 16 | data/rl-one-hour-measurements.json; rl-final-task-records.json; episode-return-points.csv; reward-points.csv; config/online-rl/ |
| S6 | 16 | data/rl-capacity-measurements.json; reported-results.csv; config/online-rl/ |
| S7 | 17 | data/reported-results.csv; Appendix A states each mechanism/replay protocol |
| S8 | 18 | data/magat-official-*; config/official-finetuning/ |
| S9 | 19 | data/magat-controlled-learning-*; config/controlled-learning/ |
| S10 | 20 | data/magat-finetune-order-* |
| Figure S1 | 21 | figures/S1_cached_learning.*; data/cached-learning-points.csv |
| Figure S2 | 21 | figures/S2_rl_learning.*; data/episode-return-points.csv; reward-points.csv |


Interpretation

- Reported gains are paired statistics where the protocol specifies paired
  aggregation; they need not equal ratios of marginal median latencies.
- S3 consumes RAM-resident raw trajectories with runtime input construction;
  it is not the fully materialized-input condition.
- S8 is a complete policy-feedback run with actual updates/corrections reported.
  S9 is a distinct identical-learning execution control. S10 holds the expert
  corpus and update budget fixed while varying sampling order.
- The one-hour PPO comparison retains native asynchronous SF scheduling.
  Its collected transitions, completed learner samples and final in-flight
  checkpoint tails are identified separately.
- S7 model-free replays measure targeted data operations, not full-training gains.
- The original fixed-budget streaming-imitation study measures online-training
  efficiency and frozen-trajectory expert-action accuracy. S4b supplies its
  separate 256-step task outcomes. S8/S10 start from released pretrained 195.pt
  and evaluate complete scene adaptation with MAPF task metrics; S9 controls
  the learning trace while comparing execution time.
- Path strings use /path/to/... placeholders. Numerical observations and task
  identifiers are preserved. Large corpora, checkpoints and runtime dependencies
  are separate experiment inputs specified by the protocols.
