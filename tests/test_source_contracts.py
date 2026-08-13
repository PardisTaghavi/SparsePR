"""Static release contracts that do not require a CUDA runtime."""

import re
import subprocess
import sys
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
    result = subprocess.run(
        ("git", "ls-files"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    paths = (
        result.stdout.splitlines()
        if result.returncode == 0
        else [path.name for path in ROOT.iterdir()]
    )
    assert not forbidden.intersection(Path(path).parts[0] for path in paths)


def test_no_research_checkout_imports() -> None:
    source = "\n".join(
        path.read_text(errors="ignore") for path in (ROOT / "src" / "sparsepr").rglob("*.py")
    )
    assert "from par." not in source
    assert re.search(r"\b(?:from|import)\s+svg(?:\.|\s|$)", source) is None


def test_adapter_package_does_not_eagerly_import_model_modules() -> None:
    subprocess.run(
        (
            sys.executable,
            "-c",
            "import sparsepr.adapters, sys; "
            "assert not any(name.startswith('sparsepr.adapters.') "
            "for name in sys.modules)",
        ),
        check=True,
    )
