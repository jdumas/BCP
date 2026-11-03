% Replay the unchanged iterationOfNewtonWithScaffold.m controller on a deterministic
% multi-step FP64 fixture and compare native Octave vs Torch evaluator trajectories.
1;
function close_worker_bridge()
    try, bcp_torch_energy('close'); catch, end
end
function [x_out, final_E, iters, cones] = run_controller_once(base_GIF, x0, numDOF, numDOF1, numDOFScaffold, totVars, almostAllVars, numR, use_torch, device)
    GIF = base_GIF;
    if use_torch
        info = bcp_torch_energy('start', device);
        assert(strcmp(info.device, device));
        GIF.useGPU = strncmp(info.device, 'cuda', 4);
        GIF.computeBackend = 'torch';
    else
        GIF.useGPU = false;
        GIF.computeBackend = 'native';
    end
    x_initial = x0;
    iter = 0;
    timesRemeshed = 0;
    stepTol1 = 1e-9;
    gradTol1 = 1e-9;
    Etol1 = 1e-9;
    EitersCounter = 0;
    numConvIters = 5;
    rate = 0;
    lambda = 0;
    areaOriginal = 1;
    terminationCondition = true;
    numFixedDOFScaffold = numDOF1;
    fprintf(1, 'BCP_REPLAY_MODE=%s\n', GIF.computeBackend);
    iterationOfNewtonWithScaffold;
    x_out = x_initial;
    final_E = GIF.NewtonFinalEnergy;
    iters = GIF.newtonIters;
    cones = GIF.UVonCones;
end

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'GIF', 'GIF', 'MatlabScripts'));
device = getenv('BCP_TORCH_REFERENCE_DEVICE');
if isempty(device), device = 'cpu'; end
cleanup = onCleanup(@close_worker_bridge);

% Construct a deterministic scaffold problem with non-degenerate |fbz| > 0 on all faces
% and outer radius large enough to run multiple Newton steps before convergence.
rng(17);
numDOF = 6;
numFixedDOFScaffold = 8;
numDOFScaffold = numFixedDOFScaffold + numDOF;
numDOF1 = numFixedDOFScaffold;
F = 40;
N = numDOFScaffold;
J_fz = 0.6 * eye(F, N) + 0.08 * (randn(F, N) + 1i * randn(F, N));
J_fbz = 0.08 * (randn(F, N) + 1i * randn(F, N));
TArea = (0.5 + rand(F, 1));
TArea = TArea / sum(TArea);

theta = linspace(0, 2 * pi, N + 1)';
theta(end) = [];
radii = [10.0 * ones(numDOF1, 1); 1.0 + 0.15 * randn(numDOF, 1)];
coords = [radii .* cos(theta); radii .* sin(theta)];
FixedIndices = [1; N + 1];
FixedValues = coords(FixedIndices);
free_idx = setdiff((1:2 * N)', FixedIndices);
x0 = coords(free_idx);
totVars = numel(x0);
almostAllVars = totVars - numDOF;

base_GIF = struct();
base_GIF.TJ_fz = J_fz;
base_GIF.TJ_fbz = J_fbz;
base_GIF.TArea = TArea;
base_GIF.FixedIndices = FixedIndices;
base_GIF.FixedValues = FixedValues;
base_GIF.ScaffoldUVonCones = x0;
for field = {'SF', 'SV', 'SJ_fz', 'SJ_fbz', 'SArea', 'EditedJ_fz', 'EditedJ_fbz'}
    base_GIF.(field{1}) = [];
end

setenv('BCP_COMPARE_TRACE_PATH', '1');
[ref_x, ref_E, ref_iters, ref_cones] = run_controller_once(base_GIF, x0, numDOF, numDOF1, numDOFScaffold, totVars, almostAllVars, F, false, device);
[cand_x, cand_E, cand_iters, cand_cones] = run_controller_once(base_GIF, x0, numDOF, numDOF1, numDOFScaffold, totVars, almostAllVars, F, true, device);
bcp_torch_energy('close');

assert(ref_iters >= 3, 'Expected multi-step Newton trajectory');
assert(ref_iters == cand_iters, 'Controller iteration count mismatch');
assert(abs(ref_E - cand_E) <= 1e-9 * max(1, abs(ref_E)), 'Controller final energy mismatch');
assert(max(abs(ref_x(:) - cand_x(:))) <= 1e-7 * max(1, max(abs(ref_x(:)))), 'Controller state mismatch');
assert(max(abs(ref_cones(:) - cand_cones(:))) <= 1e-7 * max(1, max(abs(ref_cones(:)))), 'Controller UVonCones mismatch');
fprintf('Controller replay (%s) matched reference in %d Newton steps (final E=%.12g)\n', device, cand_iters, cand_E);
