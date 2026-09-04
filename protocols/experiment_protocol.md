# Experiment Protocol

## Scope

The artifact supports the experiments reported in the Pattern Recognition
manuscript. The strongest parent-path evidence is based on paired hard and
equal-exposure blurry Split CIFAR-100 streams. Tiny ImageNet is a corrected
boundary study, and CORe50 is descriptive external-validity evidence.

## Prequential stream protocol

Each minibatch is evaluated before it is used for training. The learner does
not receive task identifiers. The hard stream has separated class experiences;
the equal-exposure blurry stream uses the registered mixing policy. Metrics are
computed from the same prequential and historical-reference interfaces across
paired methods.

## Methods

- CE-ACE: direct training parent.
- Layer 1: fixed-capacity semantic/reservoir preservation.
- Layer 2: lagged same-item deterioration-guided replay correction.
- PRBA: pre-update current/replay risk-budgeted deployment arbitration attached
  to the Layer-2 parent.
- OBC: retention-oriented comparator.

## Dataset protocols

- Split CIFAR-100: ten experiences, learning rate 0.05, memory 100, five main
  paired seeds unless a design study states otherwise.
- Tiny ImageNet D134: ten experiences, learning rate 0.05, memory 100, seeds
  218-222, with CE-ACE, Layer 2, PRBA, and OBC.
- CORe50 D135: nine experiences, learning rate 0.01, memory 100, seeds 319-321,
  with CE-ACE, Layer 1, Layer 2, PRBA, and OBC.
- CIFAR-100 D136: seeds 252-254, with random arbitration, Layer 2 without the
  Wilson gate, and PRBA without the Wilson gate.

All reported training protocols use five epochs per experience, current and
replay minibatches of 64, evaluation minibatches of 128, SGD momentum 0.9, no
weight decay, and no validation split unless explicitly stated otherwise.

## Controls and audits

The registered audits check model, memory, and replay-index hashes where
applicable; prequential order; no future, task, validation, or test leakage; no
extra replay draw; no extra backbone forward for PRBA; and no non-finite skip.
These are implementation checks, not formal causal guarantees.

## Statistical reporting

The primary paired study uses five seeds. D134 also uses five paired seeds.
D135 and D136 use three paired seeds and are reported descriptively. The
artifact preserves per-seed effects, interval estimates where computed, exact
sign-flip results where applicable, direction counts, and the stated
small-sample limitations.

## LPR status

The release includes an LPR implementation and matched CIFAR-100 runner for
reproducibility of the ongoing comparison. Production LPR results are not
included in version 1.1.0 and must not be interpreted as completed evidence.

## Dataset policy

CIFAR-100, Tiny ImageNet, and CORe50 are public datasets and are not
redistributed. Users must obtain them from their public sources and comply with
their respective terms.
