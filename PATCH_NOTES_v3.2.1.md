# CALO-RPD Studio v3.2.1 Patch Notes

## Fixed

- Corrected the Resume Center startup failure:
  `NameError: name 'manager' is not defined`.
- Signal connections now use the constructor-owned `self.manager` reference.

## Compatibility

This is a source-compatible hotfix for v3.2.0. Existing databases, experiment records, checkpoints, portfolios, and resume metadata are preserved.
