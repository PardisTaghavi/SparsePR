"""CPU-only contracts for the unified inference CLI."""

import json
from pathlib import Path

import pytest

from sparsepr.cli import (
    COSMOS3_FLOW_SHIFT,
    MODEL_IDS,
    MODEL_REVISIONS,
    _cosmos25_manifest_entry,
    _cosmos25_sparse_config,
    _wan22_sparse_options,
    build_parser,
    main,
    resolve_request,
)


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
    assert resolved["revision"] == MODEL_REVISIONS["hunyuanvideo-13b"]
    assert (
        resolved["height"],
        resolved["width"],
        resolved["frames"],
        resolved["steps"],
        resolved["fps"],
        resolved["guidance_scale"],
    ) == (720, 1280, 129, 50, 24, 6.0)


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
    assert request.size == "1280*720"
    assert (
        request.height,
        request.width,
        request.frames,
        request.steps,
        request.fps,
        request.guidance_scale,
    ) == (720, 1280, 81, 40, 16, 3.5)


def test_wan_sparse_options_match_final_policy() -> None:
    options = _wan22_sparse_options(total_layers=40, steps=40)
    assert options["first_layers_fp"] == 1
    assert options["first_sparse_forward"] == 8
    assert options["n8_selector_policy"] == "attention_mass"
    assert options["block_fusion"] is True
    assert options["cfg_layer0_reuse"] is True


def test_cosmos25_reduced_step_profile_index_is_valid() -> None:
    config = _cosmos25_sparse_config(3)
    assert config["dense_first_steps"] == 2
    assert config["profile_step"] == 2


def test_cosmos25_defaults_match_fixed_output_shape(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    image.touch()
    source = tmp_path / "cosmos-predict2.5"
    source.mkdir()
    parser = build_parser()
    args = parser.parse_args(
        (
            "--model",
            "cosmos-predict2.5-14b",
            "--prompt",
            "test prompt",
            "--image",
            str(image),
            "--output",
            str(tmp_path / "sample.mp4"),
            "--source-root",
            str(source),
        )
    )
    request = resolve_request(args, parser)
    assert (
        request.height,
        request.width,
        request.frames,
        request.steps,
        request.fps,
        request.guidance_scale,
    ) == (704, 1280, 93, 35, 16, 7.0)


def test_cosmos25_manifest_uses_requested_resolution(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    image.touch()
    source = tmp_path / "cosmos-predict2.5"
    source.mkdir()
    parser = build_parser()
    args = parser.parse_args(
        (
            "--model",
            "cosmos-predict2.5-14b",
            "--prompt",
            "test prompt",
            "--image",
            str(image),
            "--output",
            str(tmp_path / "sample.mp4"),
            "--source-root",
            str(source),
            "--height",
            "192",
            "--width",
            "320",
            "--frames",
            "93",
        )
    )
    request = resolve_request(args, parser)
    manifest = _cosmos25_manifest_entry(request)
    assert manifest["resolution"] == "192,320"


def test_cosmos3_defaults_match_paper_scheduler() -> None:
    assert COSMOS3_FLOW_SHIFT == 10.0
    parser = build_parser()
    args = parser.parse_args(
        (
            "--model",
            "cosmos3-nano-16b",
            "--prompt",
            "test prompt",
            "--image",
            __file__,
            "--output",
            "sample.mp4",
        )
    )
    request = resolve_request(args, parser)
    assert request.guidance_scale == 6.0
    assert (request.height, request.width, request.frames, request.steps, request.fps) == (
        720,
        1280,
        189,
        35,
        24,
    )
    assert request.revision == MODEL_REVISIONS["cosmos3-nano-16b"]
