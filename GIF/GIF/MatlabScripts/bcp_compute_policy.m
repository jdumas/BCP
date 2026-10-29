function [enabled, backend] = bcp_compute_policy(mode, backend)
% Keep native defaults; Octave GPU requests use the real Torch evaluator.
if nargin < 1 || isempty(mode), mode = 'auto'; end
if nargin < 2 || isempty(backend), backend = 'auto'; end
mode = lower(strtrim(mode));
backend = lower(strtrim(backend));
if ~any(strcmp(mode, {'auto', 'cpu', 'gpu'})) || ...
   ~any(strcmp(backend, {'auto', 'native', 'torch'}))
    error('BCP:InvalidComputePolicy', 'Invalid GIF device/evaluator selection.');
end
is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if strcmp(backend, 'auto')
    if is_octave && strcmp(mode, 'gpu')
        backend = 'torch';
    else
        backend = 'native';
    end
end
if strcmp(backend, 'torch')
    if ~is_octave
        error('BCP:TorchRuntime', 'Use the native MATLAB evaluator with MATLAB.');
    end
    switch mode
        case 'gpu', device = 'cuda:0';
        case 'cpu', device = 'cpu';
        otherwise, device = 'auto';
    end
    info = bcp_torch_energy('start', device);
    enabled = double(strncmp(info.device, 'cuda', 4));
    fprintf('GIF evaluator backend: torch (%s)\n', info.device);
else
    if is_octave, bcp_torch_energy('close'); end
    enabled = double(bcp_gpu_enabled(mode));
    fprintf('GIF evaluator backend: native\n');
end
end
