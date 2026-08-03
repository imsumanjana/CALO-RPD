from __future__ import annotations

from pathlib import Path


def test_native_methodology_matches_generation_level_operator_semantics():
    root = Path(__file__).resolve().parents[2]
    methodology = (root / "docs" / "calo_methodology.md").read_text(encoding="utf-8")
    normalized = " ".join(methodology.split())
    assert "native v5.9 policy samples one global operator per generation" in normalized
    assert "does not override the native neural operator action" in normalized
    assert "Each learner independently samples one of six operators" not in methodology
    assert "requires a new algorithm version" in methodology


def test_native_optimizer_preserves_the_documented_authority_boundary():
    root = Path(__file__).resolve().parents[2]
    optimizer = (root / "calo_rpd_studio" / "algorithms" / "calo" / "optimizer.py").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(optimizer.split())
    assert "raw_operator = int(decision.operator)" in normalized
    assert "raw neural operator is authoritative for ordinary learners" in normalized
    assert "they do not silently redefine the PPO" in normalized
