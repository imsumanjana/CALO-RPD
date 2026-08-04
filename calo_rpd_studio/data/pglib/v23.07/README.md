# PGLib-OPF v23.07 validation assets

This directory retains unmodified case14 assets from the official PGLib-OPF v23.07 tag at
commit `dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3`. The typical, API (binding thermal), and SAD
(small angle-difference) variants are assigned the `validation` role. They are not protected
holdouts and are not training or tuning data.

PGLib case data is licensed under Creative Commons Attribution 4.0. The upstream license is
retained as `LICENSE` with SHA-256
`95b1cd9fee1676221d74f7c0cbba622d98ac098e9b317b3848113ef4356ab4fd`. Original-source
attribution is retained in every unmodified case header. The repository software license remains
MIT; this directory's dataset is governed by the retained PGLib terms.

Each source manifest binds the official repository, release, commit, variant, role, license,
attribution, relative path, and exact asset SHA-256. Runtime import is non-executing and
fail-closed. Import creates an AC-OPF network case only; no ORPD controls are inferred. A separate
checksum-bound, reviewed ORPD profile is required before these assets can define an ORPD problem.
