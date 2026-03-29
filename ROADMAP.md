# PAE-GAN Roadmap

## Goal

Build and evaluate a Transformer-based GAN generator that progressively expands attention from local to global across stages, then compare it against fixed local and fixed global baselines on CelebA `32x32`.

## Phase 1: Project Foundation

### Objectives

- finalize a reproducible environment setup,
- establish a clean repository structure,
- define the first base experiment configuration.

### Deliverables

- portable dependency files,
- project directories for data, models, training, evaluation, configs, and outputs,
- one base config for CelebA `32x32`.

## Phase 2: Data Pipeline

### Objectives

- load images from the local `celeba/` directory,
- resize images to `32x32`,
- normalize images for GAN training,
- ensure reproducible data loading behavior.

### Tasks

- implement a CelebA dataset wrapper,
- add transform and normalization logic,
- add dataloader construction,
- add a quick sanity check for tensor shapes and sample batches.

### Exit Criteria

- a batch can be loaded successfully,
- image tensors have the expected shape and range,
- the pipeline works from config values.

## Phase 3: Training Pipeline Skeleton

### Objectives

- build the common GAN training loop before experimenting with model variants.

### Tasks

- implement hinge loss training,
- add checkpoint saving,
- add fixed latent sampling and image grid export,
- add logging for losses, timing, and memory if available,
- add seed control and run directory creation.

### Exit Criteria

- one training run can start, save outputs, and resume from a checkpoint.

## Phase 4: Baseline Models

### Objectives

- establish the minimum required baselines from the PRD.

### Tasks

- implement the CNN discriminator with spectral normalization,
- implement a stage-based Transformer generator,
- add fixed small-window attention baseline,
- add fixed global attention baseline.

### Exit Criteria

- both baselines run end to end on CelebA `32x32`,
- generated sample grids are saved during training.

## Phase 5: Progressive Attention Expansion

### Objectives

- implement the proposed method in a modular way.

### Tasks

- implement reusable `global` and `window_k` attention modules,
- support stage-wise schedules such as:
  - `["window_4", "window_4", "window_4"]`
  - `["global", "global", "global"]`
  - `["window_4", "window_8", "global"]`
- add shape checks for window partition and merge operations,
- add config-driven schedule selection.

### Exit Criteria

- the progressive model runs with configurable schedules without code changes.

## Phase 6: Stable Training And Debugging

### Objectives

- get stable low-resolution training before full experiments.

### Tasks

- run short smoke tests,
- check for collapse or instability,
- tune only the minimum needed for stable `32x32` results,
- verify that all three required model variants can train.

### Exit Criteria

- baseline local, baseline global, and progressive runs complete successfully.

## Phase 7: Evaluation

### Objectives

- measure quality and efficiency consistently across runs.

### Tasks

- implement FID computation,
- standardize generated sample export for evaluation,
- collect:
  - FID
  - training time
  - memory usage observations
  - qualitative sample grids

### Exit Criteria

- all required metrics are available for the three main variants.

## Phase 8: Ablation Study

### Objectives

- isolate the effect of the attention schedule.

### Priority Ablations

- final global stage on vs off,
- window growth patterns:
  - `4->8->global`
  - `4->8->16`
  - `8->8->global`
- number of Transformer blocks per stage,
- same schedule under different channel widths.

### Exit Criteria

- ablation results are sufficient to explain whether the schedule itself matters.

## Phase 9: Analysis And Report

### Objectives

- turn the results into a clear final project outcome.

### Deliverables

- FID comparison table,
- compute and memory comparison table,
- qualitative sample grids,
- architecture diagram,
- final report discussion covering strengths, limitations, and tradeoffs.

## Recommended Execution Order

1. Foundation and config setup
2. Data pipeline
3. Training loop skeleton
4. Discriminator and first baseline generator
5. Global baseline
6. Progressive schedule implementation
7. Main experiments
8. Ablations
9. Report and figures

## Immediate Next Tasks

1. Create the base config for CelebA `32x32`.
2. Implement the dataset loader and dataloader utility.
3. Implement the CNN discriminator.
4. Implement a minimal stage-based Transformer generator.
5. Add the first training script with checkpoint and sample saving.
