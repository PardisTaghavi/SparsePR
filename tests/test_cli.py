"""CPU-only contracts for the unified inference CLI."""

import json
from pathlib import Path

import pytest

from sparsepr.cli import MODEL_IDS, build_parser, main, resolve_request


def test_supported_model_registry_is_exact() -> None:
    assert tuple(MODEL_IDS) == (
        "hunyuanvideo-13b",
        "wan2.2-i2v-a14b",
        "cosmos-predict2.5-14b",
        "cosmos3-nano-16b",
    )


def test_hunyuan_dry_run_does_not_import_gpu_dependencies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "sample.mp4"
    main(
        (
            "--model",
            "hunyuanvideo-13b",
            "--prompt",
            "test prompt",
            "--output",
            str(output),
            "--dry-run",
        )
    )
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["model"] == "hunyuanvideo-13b"
    assert resolved["output"] == str(output)
    assert resolved["offload"] is True


def test_image_is_required_for_i2v(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        (
            "--model",
            "cosmos3-nano-16b",
            "--prompt",
            "test prompt",
            "--output",
            str(tmp_path / "sample.mp4"),
        )
    )
    with pytest.raises(SystemExit):
        resolve_request(args, parser)


def test_wan_paths_can_come_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "input.png"
    image.touch()
    source = tmp_path / "Wan2.2"
    checkpoint = tmp_path / "checkpoint"
    source.mkdir()
    checkpoint.mkdir()
    monkeypatch.setenv("SPARSEPR_WAN_ROOT", str(source))
    monkeypatch.setenv("SPARSEPR_WAN_CHECKPOINT", str(checkpoint))
    parser = build_parser()
    args = parser.parse_args(
        (
            "--model",
            "wan2.2-i2v-a14b",
            "--prompt",
            "test prompt",
            "--image",
            str(image),
            "--output",
            str(tmp_path / "sample.mp4"),
        )
    )
    request = resolve_request(args, parser)
    assert request.source_root == source
    assert request.checkpoint == checkpoint
    assert request.size == "832*480"
