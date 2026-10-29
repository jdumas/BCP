"""Standard-library-only tests: no Python geometry environment needed."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("process_utils", ROOT / "BCP/utils/process_utils.py")
process_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(process_utils)


class ProcessTests(unittest.TestCase):
    def test_literal_arguments_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix="bcp path with spaces ") as work:
            script = Path(work) / "echo args.py"
            script.write_text("import sys; print(repr(sys.argv[1:]))", encoding="utf-8")
            arguments = ["", "a path with spaces", "$(must-not-run)", "quote'"]
            result = process_utils.run_native(sys.executable, [script, *arguments], 10)
            self.assertEqual(result.stdout.strip(), repr(arguments))

    def test_failed_process_reports_diagnostics(self):
        with self.assertRaisesRegex(RuntimeError, "exited with code 7"):
            process_utils.run_native(
                sys.executable, ["-c", "import sys; print('failed'); sys.exit(7)"], 10)

    def test_timeout_is_propagated(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            process_utils.run_native(sys.executable, ["-c", "import time; time.sleep(2)"], 0.02)

    def test_thread_policy_is_inherited_not_injected(self):
        for value in (None, "7"):
            with patch.dict(os.environ):
                if value is None:
                    os.environ.pop("OMP_NUM_THREADS", None)
                else:
                    os.environ["OMP_NUM_THREADS"] = value
                result = process_utils.run_native(
                    sys.executable, ["-c", "import os; print(os.environ.get('OMP_NUM_THREADS', '<unset>'))"], 10)
            self.assertEqual(result.stdout.strip(), "<unset>" if value is None else value)

    def test_invalid_timeout_override_is_rejected(self):
        for value in ("0", "-1", "nan", "inf"):
            with patch.dict(os.environ, {"BCP_NATIVE_TIMEOUT_SECONDS": value}):
                with self.assertRaises(ValueError):
                    process_utils.run_native(sys.executable, [], 10)

    @unittest.skipUnless(os.name == "posix", "POSIX process-group cleanup")
    def test_timeout_terminates_descendants(self):
        with tempfile.TemporaryDirectory(prefix="bcp timeout tree ") as work:
            marker = Path(work)/"survived"
            ready = Path(work)/"spawned"
            child = f"import time; from pathlib import Path; time.sleep(1.5); Path({str(marker)!r}).write_text('survived')"
            parent = (
                "import subprocess,sys,time; from pathlib import Path; "
                f"subprocess.Popen([sys.executable,'-c',{child!r}], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
                f"Path({str(ready)!r}).write_text('spawned'); time.sleep(10)"
            )
            with patch.dict(os.environ):
                os.environ.pop("BCP_NATIVE_TIMEOUT_SECONDS", None)
                with self.assertRaises(subprocess.TimeoutExpired):
                    process_utils.run_native(sys.executable, ["-c", parent], 0.5)
            self.assertTrue(ready.exists(), "Test child was not spawned before the timeout")
            time.sleep(1.7)
            self.assertFalse(marker.exists(), "A descendant survived the timeout")

    def test_missing_binary_fails_promptly(self):
        with self.assertRaises(FileNotFoundError):
            process_utils.run_native(ROOT / "nonexistent-executable", [], 10)


if __name__ == "__main__":
    unittest.main()
