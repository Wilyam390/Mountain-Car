# Mountain Car project status — team note

## Where we are now

### 1) Discrete part is working
We already have a solid discrete tabular pipeline and this part can support a real section of the final notebook.

Covered so far:
- `discrete_standard`
- `discrete_fuel`
- Q-learning
- SARSA
- training/eval summaries
- policy maps
- visitation heatmaps
- value surfaces
- phase portraits / trajectories

### 2) Current discrete defaults
Use these as the working defaults unless someone improves them clearly:
- **Main discrete method:** SARSA
- **Default discretization:** `20x20`
- **Default discrete fuel coefficient:** `0.25`
- Keep **Q-learning** as the comparison baseline

### 3) Main discrete findings
- **SARSA > Q-learning** under our current setup
- `20x20` worked better than `30x30` and `40x40` with the current training budget
- In the discrete fuel variant, `0.25` is a good compromise; `0.50` hurts performance too much

## Continuous part status

### 1) Standard continuous is now working
We fixed the earlier collapse issue for the standard continuous task.

Current useful result:
- **PPO on `continuous_standard` works**
- best run reached:
  - success rate = **1.0**
  - mean steps ≈ **170**
  - mean engineered return ≈ **85.85**
- trajectories and action surface look physically sensible

This means we now have a valid **continuous standard reference result**.

### 2) Adapted continuous binary fuel is not settled yet
We tested:
- `continuous_fuel_binary`
- algorithm: **SAC**

What happened:
- **seed 0** failed and collapsed to near-inaction
- **seed 1** showed some training-side progress before the laptop died, but we do not have a finished result
- **seed 2** has not been run yet

So this setup is **not locked**. It may still fail overall, but we should finish the pending seeds before closing the door on it.

## Files we added / what they do

### Core implementation
- `src/mcrl/envs.py`
  - environment factories and wrappers
  - standard + adapted scenarios
  - consistent reward decomposition / info logging

- `src/mcrl/tabular.py`
  - discretizer
  - Q-learning
  - SARSA
  - greedy evaluation

- `src/mcrl/plots.py`
  - learning curves
  - success curves
  - policy maps
  - visitation heatmaps
  - value surfaces
  - trajectory / phase portrait plots

- `src/mcrl/continuous_utils.py`
  - continuous evaluation helpers
  - action statistics
  - trajectory extraction
  - action-surface plotting

### Execution scripts
- `scripts/run_tabular_baselines.py`
  - main discrete baseline runs

- `scripts/run_tabular_cleanup.py`
  - bin sweep
  - fuel sweep
  - final discrete comparison

- `scripts/run_continuous_baselines_v2.py`
  - current continuous runner
  - standard continuous + adapted continuous experiments
  - PPO / SAC support

### Coordination / notebook support
- `notebook_roadmap.md`
  - final notebook structure guide

- `team_project_status_note.md`
  - internal coordination note for the team

## Immediate next step for the team

The next teammate who continues should:
1. finish `continuous_fuel_binary` SAC for **seeds 1 and 2**

```bash
python scripts/run_continuous_baselines_v2.py --scenarios continuous_fuel_binary --algorithms sac --timesteps 300000 --seeds 1 2 --cost-coef 0.05 --per-step-time-cost 0.10
```

2. run `continuous_fuel_l1` SAC for **seeds 0, 1, 2**

```bash
python scripts/run_continuous_baselines_v2.py --scenarios continuous_fuel_l1 --algorithms sac --timesteps 300000 --seeds 0 1 2 --cost-coef 0.10 --per-step-time-cost 0.10
```

  Goal:
  - check whether the smoother L1 fuel cost avoids the inactivity collapse better than the binary version
  - compare adapted continuous results against the working PPO standard continuous baseline

3. summarize which adapted continuous version behaves best
4. further tune the models
5. then update this note again before we assemble the final notebook
6. do part 02 + presentation

