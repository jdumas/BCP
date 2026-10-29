"""Source-build safety/caching tests; no downloads or compiler runs in unit tests."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_octave", ROOT / "scripts" / "build_octave.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class OctaveBuildTests(unittest.TestCase):
    def archive(self, path, name, link=None):
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo(name)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = link
                archive.addfile(member)
            else:
                member.size = 4
                archive.addfile(member, io.BytesIO(b"test"))

    def test_extracts_valid_release_path(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            self.archive(root / "source.tar.gz", f"octave-{builder.VERSION}/configure")
            builder.extract_archive(root / "source.tar.gz", root / "unpack")
            self.assertEqual((root / "unpack" / f"octave-{builder.VERSION}/configure").read_text(), "test")

    def test_rejects_archive_path_traversal_and_absolute_paths(self):
        for name in ("../escape", "/escape", f"octave-{builder.VERSION}/../../escape", "other/configure"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as work:
                root = Path(work)
                self.archive(root / "source.tar.gz", name)
                with self.assertRaisesRegex(RuntimeError, "Unsafe Octave archive path"):
                    builder.extract_archive(root / "source.tar.gz", root / "unpack")
                self.assertFalse((root / "escape").exists())

    def test_rejects_escaping_archive_symlink(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            self.archive(root / "source.tar.gz", f"octave-{builder.VERSION}/link", "../../escape")
            with self.assertRaises(tarfile.FilterError):
                builder.extract_archive(root / "source.tar.gz", root / "unpack")

    def test_corrupt_cached_archive_is_not_used_or_silently_redownloaded(self):
        with tempfile.TemporaryDirectory() as work, patch.object(builder.urllib.request, "urlopen") as fetch:
            archive = Path(work) / "source.tar.gz"
            archive.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                builder.download_archive(archive)
            fetch.assert_not_called()

    def test_download_is_checked_before_publishing_cache(self):
        payload = b"downloaded source fixture"
        with tempfile.TemporaryDirectory() as work, \
             patch.object(builder.urllib.request, "urlopen", return_value=io.BytesIO(payload)), \
             patch.object(builder, "ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest()):
            archive = Path(work) / "source.tar.gz"
            builder.download_archive(archive)
            self.assertEqual(archive.read_bytes(), payload)
            self.assertFalse(list(Path(work).glob("*.part")))

    def test_bad_download_never_becomes_cached_source(self):
        with tempfile.TemporaryDirectory() as work, \
             patch.object(builder.urllib.request, "urlopen", return_value=io.BytesIO(b"corrupt")):
            archive = Path(work) / "source.tar.gz"
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                builder.download_archive(archive)
            self.assertFalse(archive.exists())
            self.assertFalse(list(Path(work).glob("*.part")))

    def test_source_build_requires_mac_and_explicit_environment(self):
        with patch.object(builder.sys, "platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "macOS-only"):
                builder.ensure_octave(ROOT, 1)
        with patch.object(builder.sys, "platform", "darwin"), \
             patch.dict(os.environ, {"PIXI_ENVIRONMENT_NAME": "octave"}):
            with self.assertRaisesRegex(RuntimeError, "no source build is selected by default"):
                builder.ensure_octave(ROOT, 1)

    def test_native_first_patch_is_apple_only_and_preserves_permission_failure(self):
        text = builder.PATCH.read_text()
        self.assertIn("+#if HAVE_MINIMALLY_WORKING_GETCWD || defined __APPLE__", text)
        self.assertIn("+# if HAVE_PARTLY_WORKING_GETCWD || defined __APPLE__", text)
        self.assertIn("errno != ERANGE && errno != ENAMETOOLONG && errno != ENOENT", text)
        self.assertNotIn("DYLD", text)

    @unittest.skipUnless(os.name == "posix", "macOS builder uses a POSIX lock")
    def test_source_build_lock_rejects_concurrent_writer_and_is_released(self):
        with tempfile.TemporaryDirectory() as work:
            base = Path(work)
            with builder.source_lock(base):
                with self.assertRaisesRegex(RuntimeError, "Another Octave source build"):
                    with builder.source_lock(base):
                        self.fail("Concurrent build acquired the lock")
            with builder.source_lock(base):
                pass

    @unittest.skipUnless(os.name == "posix", "macOS builder uses a POSIX lock")
    def test_validated_receipt_reuses_build_without_compiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"test": "validated build"}
            identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
            install = root / "build/octave-source" / identity / "install"
            executable = install / "bin/octave-cli"
            executable.parent.mkdir(parents=True)
            executable.write_text("fixture")
            executable.chmod(0o755)
            (install / "bcp-build.json").write_text(json.dumps(spec))
            with patch.object(builder.sys, "platform", "darwin"), \
                 patch.dict(os.environ, {"PIXI_ENVIRONMENT_NAME": "octave-source", "CONDA_PREFIX": directory}), \
                 patch.object(builder.shutil, "which", return_value="/fixture/tool"), \
                 patch.object(builder, "build_spec", return_value=spec), \
                 patch.object(builder, "run_logged") as run, \
                 patch.object(builder, "download_archive") as download:
                self.assertEqual(builder.ensure_octave(root, 1), executable)
                run.assert_not_called()
                download.assert_not_called()

    def test_stage_failures_invalidate_old_receipt_and_require_new_validation(self):
        for stage in ("patch", "configure", "build", "install", "smoke"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                spec = {"test": "repair build"}
                identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
                install = root / "build/octave-source" / identity / "install"
                executable = install / "bin/octave-cli"
                receipt = install / "bcp-build.json"
                install.mkdir(parents=True)
                receipt.write_text(json.dumps(spec))  # Validated once, but executable disappeared.
                failing_stage = [stage]

                def extract(archive, destination):
                    (destination / f"octave-{builder.VERSION}").mkdir()

                def run(command, cwd, env, log):
                    current = log.stem
                    if current == "install":
                        executable.parent.mkdir(exist_ok=True)
                        executable.write_text("restored interpreter")
                        executable.chmod(0o755)
                    if current == failing_stage[0]:
                        raise builder.subprocess.CalledProcessError(1, command)

                with patch.dict(os.environ, {"CONDA_PREFIX": directory}), \
                     patch.object(builder.shutil, "which", return_value="/fixture/tool"), \
                     patch.object(builder, "build_spec", return_value=spec), \
                     patch.object(builder, "download_archive"), \
                     patch.object(builder, "extract_archive", side_effect=extract), \
                     patch.object(builder, "run_logged", side_effect=run) as commands:
                    with self.assertRaises(builder.subprocess.CalledProcessError):
                        builder.build_octave(root, 1)
                    self.assertFalse(receipt.exists(), "Failed repair preserved a success receipt")
                    failing_stage[0] = None
                    commands.reset_mock()
                    self.assertEqual(builder.build_octave(root, 1), executable)
                    self.assertEqual(commands.call_args.args[-1].stem, "smoke")
                    self.assertEqual(json.loads(receipt.read_text()), spec)
                    self.assertFalse(list(install.glob("*.tmp")))

    def test_smoke_checks_real_workspace_and_sparse_solve(self):
        with tempfile.TemporaryDirectory() as work, patch.object(builder, "run_logged") as run:
            root = Path(work)
            builder.smoke(root / "octave-cli", root, {}, root / "log")
            command, cwd, env, log = run.call_args.args
            self.assertEqual(cwd, root)
            self.assertEqual(env["BCP_EXPECTED_CWD"], str(root.resolve()))
            self.assertIn("assert(strcmp(pwd()", command[-1])
            self.assertIn("sparse(", command[-1])
            self.assertIn("jsondecode(jsonencode(", command[-1])
            self.assertNotIn("cd(", command[-1])

    def test_job_count_is_bounded(self):
        self.assertEqual(builder.positive_jobs("8"), 8)
        for value in ("0", "-1", "65"):
            with self.assertRaises(builder.argparse.ArgumentTypeError):
                builder.positive_jobs(value)


if __name__ == "__main__":
    unittest.main()
