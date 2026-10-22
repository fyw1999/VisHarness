"""Exercise expert-environment setup without creating Conda envs or loading models."""

import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_IMPORT = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
NEW_IMPORT = "from torchvision.transforms.functional import rgb_to_grayscale"


def make_installation(
    tmp_path,
    content=None,
    *,
    existing=True,
    version="1.4.2",
    fail_import=None,
    working_new_api=True,
):
    environment_dir = tmp_path / "project with spaces" / "environment"
    environment_dir.mkdir(parents=True)
    script = environment_dir / "create_expert_envs.sh"
    script.write_bytes((PROJECT_ROOT / "environment/create_expert_envs.sh").read_bytes())
    for name in ("Molmo", "SAM", "SuperResolution"):
        destination = environment_dir / "experts" / f"{name}.yml"
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / "environment/experts" / f"{name}.yml").read_bytes())

    site = tmp_path / "custom site-packages"
    data = site / "basicsr" / "data"
    data.mkdir(parents=True)
    target = data / "degradations.py"
    original = content if content is not None else (OLD_IMPORT + "\n# Preserve this sentinel.\n").encode()
    target.write_bytes(original)
    target.chmod(0o640)
    (data / "__init__.py").write_text("")
    init = "from .data.degradations import rgb_to_grayscale\n"
    if fail_import == "basicsr":
        init += 'raise RuntimeError("basicsr smoke test failure")\n'
    (data.parent / "__init__.py").write_text(init)
    metadata = site / f"basicsr-{version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Name: basicsr\nVersion: {version}\n")

    transforms = site / "torchvision" / "transforms"
    transforms.mkdir(parents=True)
    (transforms.parent / "__init__.py").write_text("")
    (transforms / "__init__.py").write_text("")
    (transforms / "functional.py").write_text(
        "def rgb_to_grayscale(value):\n    return value\n" if working_new_api else ""
    )
    for name in ("gfpgan", "realesrgan"):
        (site / f"{name}.py").write_text(
            f'raise RuntimeError("{name} smoke test failure")\n' if fail_import == name else ""
        )

    state = tmp_path / "environments.json"
    state.write_text(json.dumps(["Molmo", "SAM", "SuperResolution"] if existing else []))
    log = tmp_path / "conda-calls.jsonl"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    conda = fake_bin / "conda"
    conda.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import json
            import os
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            state = Path(os.environ["EXPERT_TEST_STATE"])
            with open(os.environ["EXPERT_TEST_LOG"], "a") as output:
                output.write(json.dumps(args) + "\\n")
            names = json.loads(state.read_text())
            if args == ["env", "list"]:
                for name in names:
                    print(name, "/unused/envs/" + name)
            elif args[:3] == ["env", "create", "--file"]:
                path = Path(args[3])
                assert path.is_file(), path
                names.append(path.stem)
                state.write_text(json.dumps(names))
            elif args[:5] == ["run", "--no-capture-output", "-n", "SuperResolution", "python"]:
                assert "SuperResolution" in names
                os.environ["PYTHONPATH"] = os.environ["EXPERT_TEST_SITE"]
                os.execv(sys.executable, [sys.executable, *args[5:]])
            else:
                raise RuntimeError("Unexpected conda command: " + repr(args))
            """
        )
    )
    conda.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "EXPERT_TEST_STATE": str(state),
        "EXPERT_TEST_LOG": str(log),
        "EXPERT_TEST_SITE": str(site),
    }

    def run():
        return subprocess.run(
            ["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True
        )

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()]

    return SimpleNamespace(
        run=run,
        calls=calls,
        target=target,
        backup=target.with_name(target.name + ".visharness.bak"),
        original=original,
    )


@pytest.mark.parametrize("existing", [False, True])
def test_original_import_is_repaired_for_new_and_existing_environments(tmp_path, existing):
    installation = make_installation(tmp_path, existing=existing)
    result = installation.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert installation.target.read_bytes() == installation.original.replace(
        OLD_IMPORT.encode(), NEW_IMPORT.encode()
    )
    assert installation.backup.read_bytes() == installation.original
    assert stat.S_IMODE(installation.target.stat().st_mode) == 0o640
    assert "Applied BasicSR TorchVision import fix" in result.stdout
    assert "SuperResolution imports verified" in result.stdout
    assert result.stdout.rstrip().endswith("Visual expert environments are ready.")
    calls = installation.calls()
    creations = [call for call in calls if call[:2] == ["env", "create"]]
    assert len(creations) == (0 if existing else 3)
    assert calls[-1][:5] == ["run", "--no-capture-output", "-n", "SuperResolution", "python"]
    assert not list(installation.target.parent.glob(".visharness-basicsr-*"))


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b""])
def test_patch_preserves_line_endings_and_comments(tmp_path, newline):
    content = b"# " + OLD_IMPORT.encode() + b"\n" + OLD_IMPORT.encode() + newline
    installation = make_installation(tmp_path, content)
    result = installation.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert installation.target.read_bytes() == (
        b"# " + OLD_IMPORT.encode() + b"\n" + NEW_IMPORT.encode() + newline
    )
    assert installation.backup.read_bytes() == content


@pytest.mark.parametrize("commented_old_import", [False, True])
def test_already_compatible_environment_is_not_rewritten(tmp_path, commented_old_import):
    prefix = ("# " + OLD_IMPORT + "\n") if commented_old_import else ""
    installation = make_installation(tmp_path, (prefix + NEW_IMPORT + "\n").encode())
    before = installation.target.stat().st_mtime_ns
    result = installation.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert installation.target.read_bytes() == installation.original
    assert installation.target.stat().st_mtime_ns == before
    assert not installation.backup.exists()
    assert "already compatible" in result.stdout
    assert "SuperResolution imports verified" in result.stdout


def test_repeated_setup_preserves_original_backup_and_patched_file(tmp_path):
    installation = make_installation(tmp_path, existing=False)
    assert installation.run().returncode == 0
    after_first = installation.target.read_bytes()
    mtime = installation.target.stat().st_mtime_ns
    second = installation.run()
    assert second.returncode == 0, second.stdout + second.stderr
    assert installation.target.read_bytes() == after_first
    assert installation.target.stat().st_mtime_ns == mtime
    assert installation.backup.read_bytes() == installation.original
    assert len([call for call in installation.calls() if call[:2] == ["env", "create"]]) == 3
    assert "already compatible" in second.stdout


@pytest.mark.parametrize(
    "content",
    [
        b"# Unsupported custom file\n",
        (OLD_IMPORT + "\n" + OLD_IMPORT + "\n").encode(),
        (OLD_IMPORT + "\n" + NEW_IMPORT + "\n").encode(),
        (NEW_IMPORT + "\n" + NEW_IMPORT + "\n").encode(),
    ],
)
def test_unexpected_import_layout_fails_without_modifying_dependency(tmp_path, content):
    installation = make_installation(tmp_path, content)
    result = installation.run()
    assert result.returncode != 0
    assert "Unexpected BasicSR import layout" in result.stderr
    assert installation.target.read_bytes() == content
    assert not installation.backup.exists()
    assert "Visual expert environments are ready." not in result.stdout


def test_unexpected_basicsr_version_is_not_modified(tmp_path):
    installation = make_installation(tmp_path, version="1.4.3")
    result = installation.run()
    assert result.returncode != 0
    assert "Expected basicsr==1.4.2, found 1.4.3" in result.stderr
    assert installation.target.read_bytes() == installation.original
    assert not installation.backup.exists()
    assert "Visual expert environments are ready." not in result.stdout


def test_missing_source_file_is_not_silently_accepted(tmp_path):
    installation = make_installation(tmp_path)
    installation.target.unlink()
    result = installation.run()
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert not installation.backup.exists()
    assert "Visual expert environments are ready." not in result.stdout


def test_replacement_api_is_checked_before_modifying_source(tmp_path):
    installation = make_installation(tmp_path, working_new_api=False)
    result = installation.run()
    assert result.returncode != 0
    assert "ImportError" in result.stderr
    assert installation.target.read_bytes() == installation.original
    assert not installation.backup.exists()
    assert "Visual expert environments are ready." not in result.stdout


def test_existing_backup_is_never_overwritten(tmp_path):
    installation = make_installation(tmp_path)
    installation.backup.write_bytes(b"Earlier backup")
    result = installation.run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert installation.backup.read_bytes() == b"Earlier backup"


@pytest.mark.parametrize("module", ["basicsr", "gfpgan", "realesrgan"])
def test_failed_import_check_never_reports_environment_ready(tmp_path, module):
    installation = make_installation(tmp_path, fail_import=module)
    result = installation.run()
    assert result.returncode != 0
    assert f"{module} smoke test failure" in result.stderr
    assert "SuperResolution imports verified" not in result.stdout
    assert "Visual expert environments are ready." not in result.stdout


def test_readme_documents_weights_downloads_and_readiness():
    readme = (PROJECT_ROOT / "README.md").read_text()
    for value in (
        "realesr-general-x4v3.pth",
        "realesr-general-wdn-x4v3.pth",
        "GFPGANv1.3.pth",
        "tool_server/tool_workers/gfpgan/weights/",
        "tool_server/tool_workers/logs/server_log/",
        "all models have fully",
        "BasicSR/TorchVision compatibility fix",
    ):
        assert value in readme
