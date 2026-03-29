# PRD — Progressive Attention Expansion GAN (PAE-GAN)

## 1. Document Information

- **Project Name:** Progressive Attention Expansion GAN (PAE-GAN)
- **Project Type:** Research / Course Project
- **Prepared For:** CS485/585 Project
- **Document Version:** v1.0
- **Status:** Draft
- **Primary Goal:** Improve the quality-efficiency tradeoff in Transformer-based GAN generation by progressively expanding the attention context from local to global across generator stages.

---

## 2. Problem Statement

Transformer-based GANs are strong at modeling long-range dependencies, but using full global self-attention in every layer is computationally expensive and memory-intensive. On the other hand, fixed local or sliding-window attention is more efficient, but it may be insufficient for learning global structure and long-range consistency.

This project proposes a generator architecture in which the attention receptive field grows progressively as depth or resolution increases. The goal is to combine efficient local learning in early stages with stronger global coherence in later stages.

---

## 3. Objectives

### 3.1 Primary Objective
To investigate whether a generator with progressive attention expansion can produce better image quality and/or better global consistency than a fixed small-window attention baseline.

### 3.2 Secondary Objective
To evaluate whether the proposed design offers better computational efficiency than a full global attention generator while maintaining competitive generation quality.

### 3.3 Research Hypothesis
A generator using a local→medium→global attention schedule:
- may achieve lower FID than fixed local attention,
- and may offer a better quality-compute tradeoff than full global attention.

---

## 4. Scope

### 4.1 In Scope
- Transformer-based **generator** design
- Stage-wise control of attention type or window size
- CNN-based discriminator
- GAN training pipeline
- FID-based evaluation
- Ablation experiments
- Experiments at 32x32 resolution
- Optional 64x64 experiments if time permits

### 4.2 Out of Scope
- Proposing a new GAN loss function
- Designing a new discriminator paradigm
- Large-scale training on very large datasets
- Production deployment
- Large-scale human evaluation studies

---

## 5. Background and Motivation

In CNN-based GANs, the receptive field naturally grows with layer depth. Early layers focus on local patterns, while deeper layers capture broader structural relationships. In Transformer-based GANs, however, the attention context is often fixed: either global everywhere or local everywhere.

This project transfers the intuition of progressively growing receptive fields from CNNs to Transformer attention design. The core idea is:
- low-cost local attention in early stages,
- wider attention windows in middle stages,
- global or near-global context in final stages.

This design aims to balance efficiency and global consistency during image generation.

---

## 6. Stakeholders

### 6.1 Primary Stakeholders
- Project team
- Course instructor / evaluator
- Technical readers of the final report

### 6.2 Secondary Stakeholders
- Researchers working on GANs and image generation
- Researchers interested in efficient Transformer design

---

## 7. Success Criteria

The project will be considered successful if the following conditions are met:

1. At least 3 model variants are trained:
   - fixed small-window baseline
   - full global attention baseline
   - progressive attention schedule (proposed method)

2. Each model produces the following outputs:
   - generated image samples
   - FID score
   - training time and approximate compute cost
   - memory usage observation or measurement

3. The ablation study is sufficient to isolate and interpret the effect of the attention schedule.

4. The final report clearly explains:
   - where the method works well,
   - where it is limited,
   - and how the compute-quality tradeoff changes.

---

## 8. Functional Requirements

### 8.1 Data Pipeline
- The system must be able to load the selected dataset.
- Images must be resized to the target resolution.
- The pipeline must output normalized tensors.
- Training and evaluation runs must be reproducible.

### 8.2 Generator
- The generator must accept a latent vector `z` as input.
- The generator must project the latent vector into an initial low-resolution token or feature map.
- The generator must consist of multiple stages.
- Each stage must support configurable attention behavior.
- Attention type or window size must be independently definable per stage.
- The final stage must output an RGB image.

### 8.3 Attention Schedule
- The system must support:
  - global self-attention
  - local / window self-attention
- Each stage must allow independent attention selection.
- Example configurations:
  - `["window_4", "window_4", "window_4"]`
  - `["global", "global", "global"]`
  - `["window_4", "window_8", "global"]`

### 8.4 Discriminator
- A CNN-based discriminator must be used.
- It must be compatible with hinge loss.
- It must support spectral normalization.

### 8.5 Training Pipeline
- The system must support model training.
- Checkpoints must be saved during training.
- Fixed latent samples must be generated and saved periodically.
- Training logs must be recorded.

### 8.6 Evaluation
- The system must support FID computation.
- Different model runs must be comparable.
- Results must be reportable in both table and visual form.

---

## 9. Non-Functional Requirements

### 9.1 Reproducibility
- Random seeds must be controllable.
- Experiment runs must be config-driven.
- Identical settings must reproduce comparable runs.

### 9.2 Modularity
- The attention module must be implemented as an independent component.
- Generator and discriminator must be separated into modular files/classes.
- Adding new schedules must be easy.

### 9.3 Maintainability
- Code must be readable and documented.
- Experiment settings must be stored in config files.
- Training, evaluation, and model definitions must be separated.

### 9.4 Efficiency
- The first target is stable training at 32x32 resolution.
- Training time must fit the course project timeline.
- Memory usage must remain within available GPU limits.

---

## 10. Proposed System Design

### 10.1 High-Level Architecture
1. Latent vector `z`
2. Linear / MLP projection
3. Initial token map (e.g. `8x8xC`)
4. Stage-1 attention block(s)
5. Upsample
6. Stage-2 attention block(s)
7. Upsample
8. Stage-3 attention block(s)
9. RGB head
10. Generated image

### 10.2 Example Stage Plan
For 32x32 image generation:
- **Stage 1 (8x8):** small local window
- **Stage 2 (16x16):** medium window
- **Stage 3 (32x32):** global or very large window

### 10.3 Baseline Variants
- **Baseline A:** fixed small-window attention
- **Baseline B:** fixed global attention
- **Proposed Method:** progressive attention expansion

---

## 11. Training Design

### 11.1 Loss Functions

#### Discriminator Loss (Hinge)
`L_D = E[max(0, 1 - D(x_real))] + E[max(0, 1 + D(x_fake))]`

#### Generator Loss
`L_G = - E[D(x_fake)]`

### 11.2 Stabilization Components
- Spectral Normalization on the discriminator
- Optional R1 regularization if needed
- Consistent normalization and logging
- Small-resolution-first training strategy

---

## 12. Experiment Plan

### 12.1 Dataset
- **Primary dataset:** CelebA 32x32
- **Optional extension:** CelebA 64x64

### 12.2 Minimum Experiment Set

#### Experiment 1
- Fixed small-window attention

#### Experiment 2
- Fixed global attention

#### Experiment 3
- Progressive attention schedule

### 12.3 Ablation Ideas
- Final global stage on / off
- Window growth pattern:
  - `4→8→global`
  - `4→8→16`
  - `8→8→global`
- Number of attention blocks per stage
- Same schedule under different channel sizes

---

## 13. Metrics

### 13.1 Primary Metric
- **FID**

### 13.2 Secondary Metrics
- Training time
- GPU memory usage
- Visual sample quality
- Optional precision / recall for generative models

### 13.3 Qualitative Evaluation
- Face structure consistency
- Background coherence
- Symmetry and artifact behavior
- Long-range consistency

---

## 14. Risks and Mitigation

### 14.1 GAN Instability
**Risk:** Training collapse or mode collapse may occur.  
**Mitigation:** Use hinge loss, spectral normalization, low-resolution training first, and careful hyperparameter selection.

### 14.2 Weak Performance Gain
**Risk:** The progressive schedule may provide only marginal improvement.  
**Mitigation:** Design strong ablations, analyze the compute-quality tradeoff, and support conclusions with qualitative examples.

### 14.3 Compute Constraints
**Risk:** Full global attention may be too expensive.  
**Mitigation:** Start at 32x32, use global attention only in the final stage if needed, and allow a large-window fallback.

### 14.4 Implementation Bugs
**Risk:** Window partitioning and reshape operations may introduce bugs.  
**Mitigation:** Use modular attention design, perform shape checks on small inputs, and test tensor transformations carefully.

---

## 15. Deliverables

### 15.1 Code Deliverables
- PyTorch training code
- Generator implementation
- Discriminator implementation
- Evaluation / FID script
- Config files
- Checkpoint and sample-saving pipeline

### 15.2 Report Deliverables
- Problem definition
- Related background
- Proposed method
- Experiment setup
- Ablation study
- Quantitative results
- Qualitative results
- Discussion and limitations

### 15.3 Visual Deliverables
- Generated sample grids
- Training progress samples
- FID comparison table
- Compute comparison table
- Architecture diagram

---

## 16. Milestones

### Milestone 1 — Pipeline Setup
- Dataset loading
- Training loop skeleton
- Checkpoint and sample saving

### Milestone 2 — Baseline Model
- CNN discriminator
- One working Transformer generator baseline
- Initial stable training

### Milestone 3 — Progressive Schedule
- Window attention implementation
- Stage-wise schedule support
- Proposed model training

### Milestone 4 — Evaluation
- FID computation
- Compute / memory logging
- Sample collection

### Milestone 5 — Final Report
- Tables
- Figures
- Discussion
- Limitations and future work

---

## 17. Open Questions

- Should the final stage use full global attention or only a large window?
- How many Transformer blocks should each stage contain?
- Is a 64x64 experiment feasible within the available time?
- Which baselines can be fully trained within the compute budget?
- Should window growth be fixed or resolution-aware?

---

## 18. Final Summary

PAE-GAN proposes a stage-wise attention scheduling strategy for Transformer-based GAN generators, where the attention context gradually expands from local to global. The aim is to combine the efficiency of local attention with the structural consistency of global attention. The success of the project depends on whether this schedule produces a meaningful improvement in the quality-compute tradeoff under controlled ablation experiments.