"""Static release contracts that do not require a CUDA runtime."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_four_inference_entrypoints_exist() -> None:
    expected = {
        "hunyuanvideo_t2v.py",
        "wan22_i2v.py",
        "cosmos_predict2.py",
        "cosmos3_nano.py",
    }
    assert expected == {path.name for path in (ROOT / "examples").glob("*.py")}


def test_research_outputs_are_not_vendored() -> None:
    forbidden = {"outputs", "results", "tables", "ablations", "jobs"}
    assert not forbidden.intersection(path.name for path in ROOT.iterdir())


def test_no_research_checkout_imports() -> None:
    source = "\n".join(
        path.read_text(errors="ignore")
        for path in (ROOT / "src" / "sparsepr").rglob("*.py")
    )
    assert "from par." not in source
    assert re.search(r"\b(?:from|import)\s+svg(?:\.|\s|$)", source) is None
