# Third-Party Notices

This artifact contains or adapts components from the following projects.

## Avalanche

- Project: Avalanche
- Source: https://github.com/ContinualAI/avalanche
- License: MIT
- Copyright: Copyright (c) 2020 ContinualAI

The bundled `source/avalanche/` directory is the source snapshot used by the
experiments. Its original copyright and MIT terms are retained in `LICENSE`.

## Layerwise Proximal Replay

- Project: Layerwise Proximal Replay (LPR)
- Source: https://github.com/plai-group/LPR
- License: MIT
- Copyright: Copyright (c) 2023 Albin Soutif

`source/rbcl/lpr.py` adapts the public LPR implementation to the artifact's
Avalanche training interface. The implementation is included for the ongoing
matched comparison; production LPR results are not included in version 1.1.0.

The upstream MIT permission notice and disclaimer apply to the corresponding
adapted portions.
