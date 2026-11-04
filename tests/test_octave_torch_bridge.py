"""Dedicated tests for the in-process Octave Torch bridge (bcp_torch_energy_native).

These drive a real Octave process loading the built .oct module and compare
against the original symmetricDirichletEnergy.m in-process.  The module is
located through BCP_TORCH_NATIVE_DIR (directory containing
bcp_torch_energy_native.oct); the Octave executable comes from
BCP_OCTAVE_EXECUTABLE or PATH.  Tests skip cleanly when either is missing.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "GIF", "GIF", "MatlabScripts")
REFERENCE_TEST = os.path.join(REPO_ROOT, "tests", "torch_energy_reference.m")
NATIVE_DIR = os.environ.get("BCP_TORCH_NATIVE_DIR", "")


def octave_executable():
    return os.environ.get("BCP_OCTAVE_EXECUTABLE") or shutil.which("octave-cli") or shutil.which("octave")


def have_native_module():
    return bool(NATIVE_DIR) and os.path.isfile(os.path.join(NATIVE_DIR, "bcp_torch_energy_native.oct"))


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def run_octave(script_body, native_dir=None, device="cpu", timeout=600):
    """Run an Octave script fragment with the script path set up; fail on error."""
    octave = octave_executable()
    lines = [f"addpath('{SCRIPTS_DIR}');"]
    if native_dir:
        lines.append(f"addpath('{native_dir}');")
    lines.append(textwrap.dedent(script_body))
    with tempfile.TemporaryDirectory(prefix="bcp_bridge_test_") as tmp:
        script = os.path.join(tmp, "bridge_case.m")
        with open(script, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        env = dict(os.environ)
        env["BCP_TORCH_REFERENCE_DEVICE"] = device
        if native_dir:
            env["BCP_TORCH_NATIVE_DIR"] = native_dir
        completed = subprocess.run(
            [octave, "--no-gui", "--quiet", script],
            capture_output=True, text=True, timeout=timeout, env=env)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise AssertionError(f"Octave bridge case failed ({completed.returncode}):\n{output}")
    return output


PARITY_CASE = r"""
device = '@DEVICE@';
info = bcp_torch_energy('start', device);
assert(strcmp(info.backend, 'inprocess'));
assert(info.pid == getpid());
info_again = bcp_torch_energy('start', device);
assert(info_again.prepare_count == 0 && info_again.evaluate_count == 0);
tol = 2e-9;
check_close = @(a, b, label) assert(isempty(b) || max(abs(a(:) - b(:))) <= tol * max(1, max(abs(b(:)))), label);

rng(23);
evaluations = 0;
cases = {};
% Dense random problems exercising foldovers, fixed variables, selection.
F = 60; N = 11;
J = randn(F, N) + 1i * randn(F, N);
B = 0.2 * J;
B(1:5, :) = 3 * B(1:5, :);  % force foldovers at the test point
area = 0.1 + rand(F, 1);
x = randn(2 * N, 1);
cases{end+1} = struct('J', J, 'B', B, 'area', area, 'x', x, 'fixed', [], 'sel', []);
cases{end+1} = struct('J', J, 'B', B, 'area', area, 'x', x, 'fixed', [2; N+1], 'sel', []);
cases{end+1} = struct('J', J, 'B', B, 'area', area, 'x', x, 'fixed', [2; N+1], 'sel', [3; 1; 19]);
% Single face: fz/fbz/H1 are 1x1 (scalar ABI path).
J1 = [2 - 1i, 0.5 + 1i]; xc = [1 + 2i; 3 - 1i]; B1 = [xc(2), -xc(1)];
cases{end+1} = struct('J', J1, 'B', B1, 'area', 1.25, 'x', [real(xc); imag(xc)], 'fixed', [], 'sel', []);
cases{end+1} = struct('J', J1, 'B', B1, 'area', 1.25, 'x', [real(xc); imag(xc)], 'fixed', [], 'sel', 2);
% All faces flipped: empty fz/fbz outputs.
cases{end+1} = struct('J', J, 'B', 4 * J, 'area', area, 'x', x, 'fixed', [], 'sel', []);

for c = 1:numel(cases)
    p = cases{c};
    fixed = p.fixed;
    free = setdiff((1:columns(p.J) * 2)', fixed);
    if isempty(p.sel)
        bcp_torch_energy('prepare', p.J, p.B, p.area, fixed, p.x(fixed));
    else
        bcp_torch_energy('prepare', p.J, p.B, p.area, fixed, p.x(fixed), p.sel);
    end
    for spd = [false true]
        for include = [0 1]
            [fi, fz, fbz, E, G, H] = bcp_torch_energy('evaluate', p.x(free), spd, include);
            [fi0, fz0, fbz0, E0, G0, H0] = symmetricDirichletEnergy(p.x(free), p.J, p.B, p.area, fixed, p.x(fixed), spd, include);
            if ~isempty(p.sel), H0 = H0(p.sel, p.sel); end
            assert(isequal(fi(:), fi0(:)));  % empty foldover shapes differ upstream
            assert(isequal(size(fz), size(fz0)) && isequal(size(fbz), size(fbz0)));
            assert(isequal(size(H), size(H0)));
            check_close(fz, fz0, 'fz'); check_close(fbz, fbz0, 'fbz');
            check_close(E, E0, 'E'); check_close(G, G0, 'G'); check_close(H, H0, 'H');
            evaluations = evaluations + 1;
        end
    end
end
fprintf('parity: %d evaluations on %s\n', evaluations, device);
"""

LEDGER_CASE = r"""
info = bcp_torch_energy('start', '@DEVICE@');
rng(5);
F = 30; N = 7;
J = randn(F, N) + 1i * randn(F, N);
B = 0.25 * J;  % no foldovers at the test points: deterministic sizes
area = 0.1 + rand(F, 1);
x = randn(2 * N, 1);
fixed = [3; N + 2];
sel = [2; 5; 1];
bcp_torch_energy('prepare', J, B, area, fixed, x(fixed), sel);
st = bcp_torch_energy('stats');
assert(st.prepare_count == 1);
assert(st.matrix_upload_bytes == 2 * F * N * 16 + F * 8 + 2 * 2 * 8 + 3 * 8);
free = setdiff((1:2 * N)', fixed);
nfree = numel(free);
[fi, fz, fbz, E, G, H] = bcp_torch_energy('evaluate', x(free), true, 1);
st = bcp_torch_energy('stats');
kept = F;  % no foldovers with B = 0.25 * J
assert(isequal(size(fi), [0 1]) && isequal(size(fz), [kept 1]));
expected = 0 * 8 + 2 * kept * 16 + 8 + nfree * 8 + numel(sel)^2 * 8;
assert(st.host_download_bytes == expected, 'download ledger mismatch');
assert(st.evaluate_count == 1);
% Fewer requested outputs download fewer bytes.
[~, ~, ~, E] = bcp_torch_energy('evaluate', x(free), true, 1);
st = bcp_torch_energy('stats');
assert(st.host_download_bytes == expected + (0 * 8 + 2 * kept * 16 + 8));
% Device cache owns its copy: caller-side mutation of the inputs must not
% change evaluation results.
outs = cell(1, 6);
[outs{:}] = bcp_torch_energy('evaluate', x(free), true, 1);
first_G = outs{5}; first_H = outs{6};
J(:) = 0; area(:) = 0;
[outs{:}] = bcp_torch_energy('evaluate', x(free), true, 1);
assert(max(abs(outs{5} - first_G)) == 0);
assert(max(max(abs(outs{6} - first_H))) == 0);
% Outputs are freshly owned: later evaluations never clobber earlier results.
for step = 1:3
    bcp_torch_energy('evaluate', x(free) + 0.01 * step, true, 1);
end
assert(max(abs(outs{5} - first_G)) == 0);
assert(max(max(abs(outs{6} - first_H))) == 0);
st = bcp_torch_energy('stats');
assert(st.evaluate_count == 7 && st.prepare_count == 1);
fprintf('ledger: upload=%d download=%d bytes\n', st.matrix_upload_bytes, st.host_download_bytes);
"""

ERROR_CASE = r"""
info = bcp_torch_energy('start', '@DEVICE@');
rng(9);
J = randn(6, 3) + 1i * randn(6, 3); B = 0.3 * J; area = ones(6, 1);
bcp_torch_energy('prepare', J, B, area, [], []);
% A failing evaluate clears the evaluator rather than leaving a stale cache.
caught = false;
try, bcp_torch_energy('evaluate', 0, true, 0); catch, caught = true; end
assert(caught);
caught = false;
try, bcp_torch_energy('stats'); catch, caught = true; end
assert(caught);
caught = false;
try, bcp_torch_energy('evaluate', zeros(6, 1), true, 0); catch, caught = true; end
assert(caught);
% Restart works; close is idempotent and a fresh close is a no-op.
info = bcp_torch_energy('start', '@DEVICE@');
assert(info.prepare_count == 0 && info.evaluate_count == 0);
bcp_torch_energy('prepare', J, B, area, [], []);
bcp_torch_energy('close');
bcp_torch_energy('close');
caught = false;
try, bcp_torch_energy('stats'); catch, caught = true; end
assert(caught);
info = bcp_torch_energy('start', '@DEVICE@');
bcp_torch_energy('prepare', J, B, area, [], []);
[fi, fz, fbz, E, G, H] = bcp_torch_energy('evaluate', randn(6, 1), true, 0);
assert(isequal(size(H), [6 6]));
bcp_torch_energy('close');
fprintf('error recovery OK\n');
"""

FALLBACK_CASE = r"""
% No native module on the path: CPU reference fallback must work and CUDA
% requests must fail explicitly instead of silently changing placement.
assert(isempty(which('bcp_torch_energy_native')));
bcp_torch_energy('close');  % no-op close without any module
caught = false;
try, bcp_torch_energy('start', 'cuda:0'); catch e
    assert(strcmp(e.identifier, 'BCP:TorchRuntime')); caught = true;
end
assert(caught, 'CUDA request without the native module must fail explicitly');
info = bcp_torch_energy('start', 'cpu');
assert(strcmp(info.backend, 'octave-reference') && strcmp(info.device, 'cpu'));
assert(info.pid == getpid());
rng(17);
F = 25; N = 6;
J = randn(F, N) + 1i * randn(F, N); B = 0.3 * J; area = 0.2 + rand(F, 1);
x = randn(2 * N, 1); fixed = [1; N + 2]; sel = [2; 4];
free = setdiff((1:2 * N)', fixed);
bcp_torch_energy('prepare', J, B, area, fixed, x(fixed), sel);
for spd = [false true]
    for include = [0 1]
        [fi, fz, fbz, E, G, H] = bcp_torch_energy('evaluate', x(free), spd, include);
        [fi0, fz0, fbz0, E0, G0, H0] = symmetricDirichletEnergy(x(free), J, B, area, fixed, x(fixed), spd, include);
        assert(isequal(H, H0(sel, sel)));
        assert(isequal(fi, fi0) && isequal(G, G0) && isequal(E, E0));
    end
end
st = bcp_torch_energy('stats');
assert(st.prepare_count == 1 && st.evaluate_count == 4);
assert(st.matrix_upload_bytes == 2 * F * N * 16 + F * 8 + 2 * 2 * 8 + 2 * 8);
bcp_torch_energy('close');
fprintf('fallback reference OK\n');
"""


def bridge_unavailable_reason():
    if octave_executable() is None:
        return "no Octave executable"
    if not have_native_module():
        return "BCP_TORCH_NATIVE_DIR does not contain bcp_torch_energy_native.oct"
    return None


@unittest.skipIf(bridge_unavailable_reason() is not None, bridge_unavailable_reason() or "")
class OctaveTorchBridgeNativeTest(unittest.TestCase):
    def test_native_cpu_parity(self):
        run_octave(PARITY_CASE.replace("@DEVICE@", "cpu"), native_dir=NATIVE_DIR)

    @unittest.skipUnless(cuda_available(), "CUDA is not available")
    def test_native_cuda_parity(self):
        run_octave(PARITY_CASE.replace("@DEVICE@", "cuda:0"), native_dir=NATIVE_DIR)

    def test_native_stats_and_copy_ledger_cpu(self):
        run_octave(LEDGER_CASE.replace("@DEVICE@", "cpu"), native_dir=NATIVE_DIR)

    @unittest.skipUnless(cuda_available(), "CUDA is not available")
    def test_native_stats_and_copy_ledger_cuda(self):
        run_octave(LEDGER_CASE.replace("@DEVICE@", "cuda:0"), native_dir=NATIVE_DIR)

    def test_native_error_recovery_cpu(self):
        run_octave(ERROR_CASE.replace("@DEVICE@", "cpu"), native_dir=NATIVE_DIR)

    @unittest.skipUnless(cuda_available(), "CUDA is not available")
    def test_native_error_recovery_cuda(self):
        run_octave(ERROR_CASE.replace("@DEVICE@", "cuda:0"), native_dir=NATIVE_DIR)

    def test_reference_harness_cpu_through_native_module(self):
        # The shared 65+ comparison suite, exercised through the native bridge.
        env_device = "cpu"
        with tempfile.TemporaryDirectory(prefix="bcp_bridge_ref_"):
            run_octave("run('%s');" % REFERENCE_TEST, native_dir=NATIVE_DIR, device=env_device)

    @unittest.skipUnless(cuda_available(), "CUDA is not available")
    def test_reference_harness_cuda_through_native_module(self):
        run_octave("run('%s');" % REFERENCE_TEST, native_dir=NATIVE_DIR, device="cuda:0")


@unittest.skipIf(octave_executable() is None, "no Octave executable")
class OctaveTorchBridgeFallbackTest(unittest.TestCase):
    def test_cpu_reference_fallback_without_native_module(self):
        run_octave(FALLBACK_CASE, native_dir=None)


if __name__ == "__main__":
    unittest.main()
