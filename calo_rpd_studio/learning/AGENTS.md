# Learning instructions

- Learning/training is an independent lifecycle with its own configuration and checkpoints.
- Training may create candidates but must never auto-qualify or auto-activate them.
- Exclude protected identities and derivatives from every learning, tuning, and selection path.
- Phase 4 performs no policy training or evaluation. It may complete training infrastructure and
  empty-policy behavior using deterministic synthetic fixtures only.
- After the development freeze, train a completely new policy without old weights or activation/
  qualification state; old policies are not final candidates or initializers.
