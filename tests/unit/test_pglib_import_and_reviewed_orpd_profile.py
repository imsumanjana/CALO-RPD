from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from calo_rpd_studio.orpd.external_profile import (
    ReviewedORPDProfile,
    ReviewedORPDProfileError,
    load_reviewed_orpd_profile,
    variable_config_from_reviewed_profile,
)
from calo_rpd_studio.orpd.formulation_fingerprint import scientific_problem_payload
from calo_rpd_studio.orpd.problem import ORPDProblem, ORPDProblemConfig
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableDecoder, ShuntControlDefinition
from calo_rpd_studio.power_system.pglib_import import (
    PGLibImportError,
    PGLibSourceManifest,
    available_bundled_pglib_cases,
    import_pglib_case,
    load_bundled_pglib_case,
)


CASE_NAME = "pglib_opf_case3_calo_fixture"
CASE_SOURCE = f"""% PGLib-OPF test-shaped fixture; not external evidence
function mpc = {CASE_NAME}
mpc.version = '2';
mpc.baseMVA = 100;
mpc.areas = [
1 1;
];
mpc.bus = [
1 3 0 0 0 0 1 1.04 0 230 1 1.10 0.90;
2 2 20 10 0 0 1 1.01 0 230 1 1.10 0.90;
3 1 45 15 0 0 1 1.00 0 230 1 1.10 0.90;
];
mpc.gen = [
1 40 0 100 -100 1.04 100 1 200 0;
2 30 0 100 -100 1.01 100 1 150 0;
];
mpc.branch = [
1 2 0.02 0.06 0.03 200 200 200 0 0 1 -360 360;
1 3 0.08 0.24 0.025 200 200 200 0 0 1 -360 360;
2 3 0.06 0.18 0.02 200 200 200 1.0 0 1 -360 360;
];
mpc.gencost = [
2 0 0 3 0.02 2 0;
2 0 0 3 0.03 1 0;
];
"""


def _write_case(tmp_path, source: str = CASE_SOURCE):
    path = tmp_path / f"{CASE_NAME}.m"
    raw = source.encode("utf-8")
    path.write_bytes(raw)
    manifest = PGLibSourceManifest(
        release_tag="v23.07",
        source_commit="a" * 40,
        relative_path=path.name,
        variant="typical",
        asset_sha256=hashlib.sha256(raw).hexdigest(),
        case_role="validation",
        attribution="PGLib-OPF contributors; fixture structure only",
    )
    return path, manifest


def _profile(case, manifest) -> ReviewedORPDProfile:
    return ReviewedORPDProfile(
        profile_id="fixture-reviewed-orpd",
        profile_version="1.0.0",
        review_status="reviewed",
        reviewed_by="CALO-RPD test reviewer",
        reviewed_at_utc="2026-08-04T00:00:00Z",
        review_evidence="tests/unit fixture review",
        rationale="Explicit fixture controls exercise the import/formulation boundary.",
        source_asset_sha256=manifest.asset_sha256,
        physical_case_checksum=case.checksum(),
        generator_voltage_buses=(2,),
        transformer_branch_indices=(2,),
        shunt_controls=(ShuntControlDefinition(3, 0.0, 5.0, 1.0, "absolute", "reviewed fixture"),),
    )


def test_verified_pglib_import_retains_provenance_without_inferring_orpd(tmp_path):
    path, manifest = _write_case(tmp_path)
    case = import_pglib_case(path, manifest)

    assert case.name == CASE_NAME
    assert case.n_bus == 3
    assert case.source_provenance["asset_sha256"] == manifest.asset_sha256
    assert case.source_provenance["case_role"] == "validation"
    assert case.source_provenance["physical_case_checksum"] == case.checksum()
    assert case.source_provenance["ignored_matpower_fields"] == ["areas"]
    assert "no ORPD controls inferred" in case.source_provenance["import_semantics"]
    assert case.clone().source_provenance == case.source_provenance
    assert type(case).from_dict(case.to_dict()).source_provenance == case.source_provenance


def test_bundled_official_typical_api_and_sad_assets_verify_and_load():
    expected = {
        "pglib-case14-typical": "14488f24f83576bfb80179434f27ae036ccec4e3ba69000cb3ec45ed8d3376d2",
        "pglib-case14-api": "d845a7205808a982edac7cccae71b768872bdc5b76ad212e3bc116af45a1c421",
        "pglib-case14-sad": "e2e92121499920b0048c920d9db7c2375e2014d7b5f96c7e0f04f43c0d56e62e",
    }
    assert available_bundled_pglib_cases() == tuple(expected)
    loaded = {name: load_bundled_pglib_case(name) for name in expected}

    for name, case in loaded.items():
        assert (case.n_bus, case.n_gen, case.n_branch) == (14, 5, 20)
        assert case.checksum() == expected[name]
        assert case.source_provenance["source_commit"] == (
            "dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3"
        )
        assert case.source_provenance["case_role"] == "validation"
    assert len({case.checksum() for case in loaded.values()}) == 3

    with pytest.raises(PGLibImportError, match="Unknown bundled"):
        load_bundled_pglib_case("pglib-case118-protected")


def test_pglib_import_rejects_hash_syntax_variant_and_closed_protected_asset(tmp_path):
    path, manifest = _write_case(tmp_path)
    with pytest.raises(PGLibImportError, match="SHA-256 mismatch"):
        import_pglib_case(path, replace(manifest, asset_sha256="0" * 64))

    malicious = CASE_SOURCE + "system('forbidden');\n"
    malicious_path, malicious_manifest = _write_case(tmp_path, malicious)
    with pytest.raises(PGLibImportError, match="unsupported MATLAB syntax"):
        import_pglib_case(malicious_path, malicious_manifest)

    with pytest.raises(PGLibImportError, match="declared variant"):
        replace(manifest, variant="api").validate()
    with pytest.raises(PGLibImportError, match="must be a string"):
        replace(manifest, source_commit=123).validate()

    path, manifest = _write_case(tmp_path)
    protected = replace(manifest, case_role="protected_test")
    with pytest.raises(PGLibImportError, match="explicit test-only access"):
        import_pglib_case(path, protected)
    assert import_pglib_case(path, protected, allow_protected_test=True).n_bus == 3


def test_reviewed_profile_is_checksum_bound_and_selects_only_declared_controls(tmp_path):
    path, manifest = _write_case(tmp_path)
    case = import_pglib_case(path, manifest)
    profile = _profile(case, manifest)
    config = variable_config_from_reviewed_profile(case, profile)
    decoder = ORPDVariableDecoder(case, config)

    assert [variable.name for variable in decoder.variables] == ["Vg@2", "Tap 2-3", "Qsh@3"]
    assert profile.checksum() in config.formulation_profile
    problem = ORPDProblem(case, ORPDProblemConfig(variables=config))
    payload = scientific_problem_payload(problem)
    assert payload["case_source_provenance"]["source_commit"] == "a" * 40
    assert payload["formulation_manifest"]["generator_voltage_buses"] == [2]

    with pytest.raises(ReviewedORPDProfileError, match="source asset"):
        variable_config_from_reviewed_profile(case, replace(profile, source_asset_sha256="0" * 64))
    with pytest.raises(ReviewedORPDProfileError, match="physical checksum"):
        variable_config_from_reviewed_profile(
            case, replace(profile, physical_case_checksum="0" * 64)
        )
    with pytest.raises(ReviewedORPDProfileError, match="JSON booleans"):
        replace(profile, discrete_shunts="false")


def test_reviewed_profile_json_requires_exact_asset_hash_and_schema(tmp_path):
    case_path, manifest = _write_case(tmp_path)
    case = import_pglib_case(case_path, manifest)
    profile = _profile(case, manifest)
    profile_path = tmp_path / "reviewed-profile.json"
    raw = (json.dumps(profile.payload(), sort_keys=True) + "\n").encode("utf-8")
    profile_path.write_bytes(raw)

    loaded = load_reviewed_orpd_profile(
        profile_path, expected_sha256=hashlib.sha256(raw).hexdigest()
    )
    assert loaded == profile
    with pytest.raises(ReviewedORPDProfileError, match="asset SHA-256 mismatch"):
        load_reviewed_orpd_profile(profile_path, expected_sha256="0" * 64)

    unknown = profile.payload()
    unknown["auto_infer_controls"] = True
    unknown_raw = json.dumps(unknown).encode("utf-8")
    profile_path.write_bytes(unknown_raw)
    with pytest.raises(ReviewedORPDProfileError, match="exactly match"):
        load_reviewed_orpd_profile(
            profile_path, expected_sha256=hashlib.sha256(unknown_raw).hexdigest()
        )


def test_reviewed_profile_preserves_protected_case_boundary(tmp_path):
    path, manifest = _write_case(tmp_path)
    protected_manifest = replace(manifest, case_role="protected_test")
    case = import_pglib_case(path, protected_manifest, allow_protected_test=True)
    profile = _profile(case, protected_manifest)

    with pytest.raises(ReviewedORPDProfileError, match="explicit test-only access"):
        variable_config_from_reviewed_profile(case, profile)
    config = variable_config_from_reviewed_profile(case, profile, allow_protected_test=True)
    assert ORPDVariableDecoder(case, config).dimension == 3
