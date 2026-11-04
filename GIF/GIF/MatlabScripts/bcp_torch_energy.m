function varargout = bcp_torch_energy(operation, varargin)
% In-process bridge to the analytic Torch evaluator (Octave only).
%
% Preferred backend is the bcp_torch_energy_native .oct module, which embeds
% CPython in this process and reuses BCP/GIF_and_CM/torch_energy.py directly:
% Jacobians are uploaded to the Torch device once per prepare/remesh, each
% evaluate uploads only the free coordinates, and device results are written
% straight into Octave-owned output arrays.  There is no worker process and
% no serialization.
%
% If the native module is not on the path, an explicit 'cpu' (or 'auto')
% device falls back to the original symmetricDirichletEnergy.m evaluated
% in-process as the CPU reference; requesting CUDA without the module fails
% explicitly instead of silently changing placement.
%
% Operations: 'start' device, 'prepare' J_fz J_fbz area fixed_indices
% fixed_values [hessian_indices], 'evaluate' x spd include_flips_flag
% (0 excludes foldover faces), 'stats', 'close'.  'close' with no live
% evaluator is a no-op that never initializes Python or CUDA.
persistent state;
if exist('OCTAVE_VERSION', 'builtin') == 0
    error('BCP:TorchRuntime', 'The Torch bridge currently requires Octave.');
end
try
    switch operation
        case 'start'
            requested = varargin{1};
            if ~isempty(state) && strcmp(state.requested, requested)
                varargout{1} = state.meta;
                return;
            end
            state = [];
            if ~isempty(which('bcp_torch_energy_native'))
                meta = bcp_torch_energy_native('start', requested, torch_module_dir());
                state = struct('requested', requested, 'backend', 'native', ...
                               'meta', meta, 'prepared', []);
            elseif any(strcmp(requested, {'cpu', 'auto'}))
                if ~strcmp(requested, 'cpu')
                    fprintf(['bcp_torch_energy: native module unavailable; ' ...
                             'using the in-process Octave CPU reference (auto -> cpu).\n']);
                end
                state = struct('requested', requested, 'backend', 'reference', ...
                               'meta', reference_meta(), 'prepared', []);
            else
                error('BCP:TorchRuntime', ['The native module bcp_torch_energy_native ' ...
                      'is not on the Octave path; cannot start a CUDA Torch evaluator. ' ...
                      'Load the module or request the CPU reference.']);
            end
            varargout{1} = state.meta;
        case 'close'
            if ~isempty(state) && strcmp(state.backend, 'native')
                bcp_torch_energy_native('close');
            end
            state = [];
        case 'prepare'
            require_connection(state);
            fixed = varargin{4}(:);
            values = varargin{5}(:);
            upload = array_bytes(varargin{1}) + array_bytes(varargin{2}) + ...
                     array_bytes(varargin{3}(:)) + array_bytes(fixed) + array_bytes(values);
            has_selection = numel(varargin) >= 6;
            selection = [];
            if has_selection
                selection = varargin{6}(:);
                upload = upload + array_bytes(selection);
            end
            if strcmp(state.backend, 'native')
                % Only the device cache is needed after upload. Do not retain
                % the previous scaffold's host Jacobians across remeshing.
                state.prepared = [];
                if has_selection
                    bcp_torch_energy_native('prepare', varargin{1}, varargin{2}, ...
                                            varargin{3}(:), fixed, values, selection);
                else
                    bcp_torch_energy_native('prepare', varargin{1}, varargin{2}, ...
                                            varargin{3}(:), fixed, values);
                end
                state.prepared = true;
            else
                state.prepared = struct('J_fz', varargin{1}, 'J_fbz', varargin{2}, ...
                    'area', varargin{3}(:), 'fixed', fixed, 'values', values, ...
                    'has_selection', has_selection, 'selection', selection);
            end
            state.meta.prepare_count = state.meta.prepare_count + 1;
            state.meta.matrix_upload_bytes = state.meta.matrix_upload_bytes + upload;
        case 'evaluate'
            require_connection(state);
            if isempty(state.prepared)
                error('BCP:TorchProcess', 'The Torch evaluator must be prepared before evaluate.');
            end
            x = varargin{1}(:);
            spd = logical(varargin{2});
            exclude_flips = varargin{3} == 0;
            if strcmp(state.backend, 'native')
                outputs = cell(1, max(nargout, 1));
                [outputs{:}] = bcp_torch_energy_native('evaluate', x, spd, exclude_flips);
                varargout = outputs(1:nargout);
            else
                p = state.prepared;
                [fi, fz, fbz, E, G, H] = symmetricDirichletEnergy( ...
                    x, p.J_fz, p.J_fbz, p.area, p.fixed, p.values, spd, varargin{3});
                if p.has_selection
                    H = H(p.selection, p.selection);
                end
                results = {fi, fz, fbz, E, G, H};
                varargout = results(1:nargout);
                downloaded = 0;
                for k = 1:nargout
                    downloaded = downloaded + array_bytes(results{k});
                end
                state.meta.host_download_bytes = state.meta.host_download_bytes + downloaded;
            end
            state.meta.evaluate_count = state.meta.evaluate_count + 1;
        case 'stats'
            require_connection(state);
            if strcmp(state.backend, 'native')
                state.meta = bcp_torch_energy_native('stats');
            end
            info = whos('state');
            meta = state.meta;
            meta.dispatcher_state_bytes = info.bytes;
            varargout{1} = meta;
        otherwise
            error('BCP:TorchProtocol', 'Unknown Torch evaluator operation: %s', operation);
    end
catch exception
    if ~isempty(state) && strcmp(state.backend, 'native')
        try, bcp_torch_energy_native('close'); catch, end
    end
    state = [];
    rethrow(exception);
end
end

function require_connection(state)
if isempty(state)
    error('BCP:TorchProcess', 'The Torch evaluator must be started before prepare/evaluate.');
end
end

function bytes = array_bytes(value)
bytes = numel(value) * 8 * (2 - double(isreal(value)));
end

function meta = reference_meta()
meta = struct('device', 'cpu', 'backend', 'octave-reference', 'pid', getpid(), ...
              'prepare_count', 0, 'evaluate_count', 0, 'matrix_upload_bytes', 0, ...
              'host_download_bytes', 0);
end

function dir = torch_module_dir()
override = getenv('BCP_TORCH_MODULE_DIR');
if ~isempty(override)
    dir = override;
else
    dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', ...
                   'BCP', 'GIF_and_CM');
end
end
