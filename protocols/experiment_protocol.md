# Experiment Protocol

## Scope

The artifact supports the experiments reported in the Pattern Recognition
manuscript. The primary causal claim is restricted to paired hard and blurry
Split CIFAR-100 streams. Tiny ImageNet is used as boundary and transfer
evidence, not as proof of universal dominance.

## Stream protocol

The learner receives a ten-experience online class-incremental stream without
task identifiers. Each minibatch is evaluated before it is used for training.
The hard stream has separated class experiences. The equal-exposure blurry
stream exposes classes with the registered equal-exposure mixing policy.

## Methods

- Base: CE-ACE direct training parent.
- Layer 2: fixed-capacity hybrid memory plus lagged same-item deterioration-
  guided replay correction.
- Layer 3: PRBA prequential risk-budgeted arbitration attached to the parent.
- Full: Layer 2 followed by Layer 3 with the same audited parent path.

## Controls and audits

The registered audits check model, memory, and replay-index hashes where
applicable; prequential order; no future, task, validation, or test leakage;
no extra replay draw; no extra backbone forward for PRBA; and no non-finite
skip. The audit is an implementation check, not a formal causal guarantee.

## Statistical reporting

The primary D133 comparison uses five paired seeds. The artifact reports the
raw paired effects, exhaustive percentile bootstrap intervals over all 5^5
resamples, exact sign-flip tests over all 2^5 sign assignments, Cohen's d_z,
and direction counts. These are small-sample evidence and must not be
described as large-sample statistical significance.

## Dataset policy

CIFAR-100 and Tiny ImageNet are public datasets and are not redistributed in
this artifact. Users must obtain them from their public sources and comply
with the respective dataset terms.
