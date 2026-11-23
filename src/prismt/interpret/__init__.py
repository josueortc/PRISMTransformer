"""PRISMt interpretation: attribution of each prediction onto channel x time x modality,
layer-head reduction, and non-negative CP decomposition into motifs.

The modules follow the order of the method:

* :mod:`attribution` -- G[n, layer, head, token] = rollout routing x signed value message
* :mod:`reduce` -- class-discriminative head weights; per-trial and discriminative tensors
* :mod:`cp` -- non-negative CP with the modality factor on the simplex
* :mod:`align` -- matching motifs across folds and resamples
* :mod:`stats` -- bootstrap intervals and label-permutation significance
* :mod:`faithfulness` -- occlusion and interchange interventions
* :mod:`baselines` -- the methods PRISMt is compared with, as maps on the same axes
* :mod:`metrics` -- recovery scores against known ground truth (synthetic data)
* :mod:`pipeline` -- all of the above for a finished run folder
"""
