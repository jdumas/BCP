function enabled = bcp_gpu_enabled(mode)
% Select the existing gpuArray path; never emulate GPU computation on the CPU.
if nargin == 0 || isempty(mode)
    mode = 'auto';
end
mode = lower(strtrim(mode));
if ~any(strcmp(mode, {'auto', 'cpu', 'gpu'}))
    error('BCP:InvalidGPUDevice', 'GIF device must be auto, cpu, or gpu.');
end
enabled = false;
if strcmp(mode, 'cpu')
    return;
end
if exist('OCTAVE_VERSION', 'builtin') == 0
    try
        if exist('canUseGPU', 'file') ~= 0 || exist('canUseGPU', 'builtin') ~= 0
            enabled = canUseGPU;
        elseif exist('gpuDeviceCount', 'file') ~= 0
            enabled = gpuDeviceCount > 0;
        end
    catch exception
        if strcmp(mode, 'gpu')
            rethrow(exception);
        end
        warning('BCP:GPUUnavailable', 'GIF is using CPU: %s', exception.message);
    end
end
if strcmp(mode, 'gpu') && ~enabled
    error('BCP:GPUUnavailable', ...
        'Native GIF GPU requested, but unavailable. Requires MATLAB, Parallel Computing Toolbox and a working CUDA GPU; select the Torch evaluator for Octave GPU execution.');
end
end
