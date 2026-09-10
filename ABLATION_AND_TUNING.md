# SafeGrip-CI ablation and hyperparameter protocol

## Controlled primary ablation

All primary variants use identical split definitions, endpoint identities, seeds, and the same fixed full-model hyperparameters unless the optional retuned-ablation study is requested.

The primary set is:

- `safegrip_backbone_raw`
- `safegrip_persistent`
- `safegrip_neural_innovation`
- `safegrip_no_identifiability`
- `safegrip_excitation_proxy`
- `safegrip_no_acceptance`
- `safegrip_no_innovation_supervision`
- `safegrip_no_bound`
- `safegrip_no_uq`
- `safegrip`

The scientifically central comparison is `safegrip_excitation_proxy` versus `safegrip`: it tests whether counterfactual friction identifiability is more useful than a conventional handcrafted excitation proxy.

The ablation registry is explicit rather than inferred from names. Tests require every primary variant to have distinct semantics.

## Point objective

The full model is trained with Huber point error, direct latent innovation supervision, friction-conditioned dynamics reconstruction, counterfactual ranking, and a do-no-harm update penalty. UQ is fitted only after point-model selection.

## Hyperparameter search

Validation-only tuning may search:

- sequence length and encoder sizes;
- evidence window;
- innovation scale;
- state persistence;
- counterfactual friction displacement;
- identifiability scale;
- acceptance temperature;
- innovation/dynamics/counterfactual/do-no-harm loss weights;
- post-hoc information-conditioned UQ inflation.

The test set is locked during search. Proposal and literature-comparator tuning preserve the common endpoint-hash parity checks already used by the repository.
