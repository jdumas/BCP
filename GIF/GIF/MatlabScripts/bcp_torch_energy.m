function varargout = bcp_torch_energy(operation, varargin)
% Persistent binary-pipe bridge to the analytic Torch evaluator (Octave only).
% Jacobians are uploaded once per prepare/remesh, not once per Newton step.
persistent connection;
if exist('OCTAVE_VERSION', 'builtin') == 0
    error('BCP:TorchRuntime', 'The Torch pipe backend currently requires Octave.');
end
try
    switch operation
        case 'start'
            requested = varargin{1};
            if ~isempty(connection) && strcmp(connection.requested, requested)
                varargout{1} = connection.meta;
                return;
            end
            connection = [];
            executable = getenv('BCP_PYTHON_EXECUTABLE');
            if isempty(executable), executable = 'python3'; end
            worker = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', ...
                              'BCP', 'GIF_and_CM', 'torch_energy_worker.py');
            [input, output, pid] = popen2(executable, {worker});
            if pid < 0, error('BCP:TorchProcess', 'Cannot start the Torch evaluator.'); end
            state = struct('input', input, 'output', output, 'pid', pid, ...
                           'requested', requested);
            state.cleanup = onCleanup(@() close_worker(input, output, pid));
            [~, state.meta] = request_worker(state, struct('op', 'start', 'device', requested), struct());
            connection = state;
            varargout{1} = connection.meta;
        case 'close'
            connection = [];
        case 'prepare'
            require_connection(connection);
            arrays = struct('J_fz', varargin{1}, 'J_fbz', varargin{2}, ...
                            'area', varargin{3}(:), 'fixed_indices', varargin{4}(:), ...
                            'fixed_values', varargin{5}(:));
            if numel(varargin) >= 6, arrays.hessian_indices = varargin{6}(:); end
            [~, connection.meta] = request_worker(connection, struct('op', 'prepare'), arrays);
        case 'evaluate'
            require_connection(connection);
            header = struct('op', 'evaluate', 'spd', logical(varargin{2}), ...
                            'exclude_flips', varargin{3} == 0);
            [result, connection.meta] = request_worker(connection, header, struct('x', varargin{1}(:)));
            names = {'foldovers', 'fz', 'fbz', 'E', 'G', 'H'};
            for k = 1:nargout, varargout{k} = result.(names{k}); end
        case 'stats'
            require_connection(connection);
            [~, connection.meta] = request_worker(connection, struct('op', 'stats'), struct());
            varargout{1} = connection.meta;
        otherwise
            error('BCP:TorchProtocol', 'Unknown Torch evaluator operation: %s', operation);
    end
catch exception
    connection = [];
    rethrow(exception);
end
end

function require_connection(connection)
if isempty(connection)
    error('BCP:TorchProcess', 'The Torch evaluator must be started before prepare/evaluate.');
end
end

function [result, meta] = request_worker(state, header, arrays)
names = fieldnames(arrays);
descriptors = cell(1, numel(names));
for k = 1:numel(names)
    value = arrays.(names{k});
    if ndims(value) ~= 2, error('BCP:TorchProtocol', 'Expected a matrix.'); end
    descriptors{k} = struct('name', names{k}, 'rows', rows(value), ...
                            'cols', columns(value), 'complex', ~isreal(value));
end
header.arrays = descriptors;
if fputs(state.input, [jsonencode(header), sprintf('\n')]) < 0 || fflush(state.input) ~= 0
    error('BCP:TorchProtocol', 'Cannot write Torch request header.');
end
for k = 1:numel(names)
    value = arrays.(names{k});
    write_doubles(state.input, real(value(:)));
    if ~isreal(value), write_doubles(state.input, imag(value(:))); end
end
if fflush(state.input) ~= 0, error('BCP:TorchProtocol', 'Cannot flush Torch request.'); end
response = jsondecode(read_header(state.output));
if ~isfield(response, 'ok') || ~response.ok
    if isfield(response, 'error'), detail = response.error; else, detail = 'Malformed response'; end
    error('BCP:TorchEvaluator', '%s', detail);
end
result = struct();
for k = 1:numel(response.arrays)
    item = response.arrays(k);
    count = item.rows * item.cols;
    value = read_doubles(state.output, count);
    if item.complex, value = complex(value, read_doubles(state.output, count)); end
    result.(item.name) = reshape(value, item.rows, item.cols);
end
meta = response.meta;
end

function write_doubles(stream, value)
if fwrite(stream, double(value), 'double', 0, 'ieee-le') ~= numel(value)
    error('BCP:TorchProtocol', 'Truncated Torch request payload.');
end
end

function seconds = timeout_seconds()
seconds = str2double(getenv('BCP_TORCH_TIMEOUT_SECONDS'));
if ~isfinite(seconds) || seconds <= 0, seconds = 600; end
end

function value = read_header(stream)
value = '';
started = tic();
while true
    fclear(stream); errno(0);
    part = fgets(stream);
    blocked = errno() == errno('EAGAIN') || errno() == errno('EWOULDBLOCK');
    if ischar(part)
        value = [value, part];
        if numel(value) > 65536, error('BCP:TorchProtocol', 'Oversized worker response.'); end
        if ~isempty(value) && value(end) == char(10), return; end
    elseif ~blocked && feof(stream)
        error('BCP:TorchProtocol', 'Torch worker closed its output before completing a response.');
    else
        fclear(stream);
        if toc(started) > timeout_seconds(), error('BCP:TorchTimeout', 'Torch response timed out.'); end
        pause(0.001);
    end
end
end

function value = read_doubles(stream, count)
% Read bytes first: a nonblocking read may split a double between chunks.
raw = zeros(8 * count, 1, 'uint8');
offset = 0;
started = tic();
while offset < numel(raw)
    fclear(stream); errno(0);
    part = fread(stream, min(numel(raw) - offset, 65536), '*uint8');
    blocked = errno() == errno('EAGAIN') || errno() == errno('EWOULDBLOCK');
    if ~isempty(part)
        raw(offset + (1:numel(part))) = part;
        offset = offset + numel(part);
    elseif ~blocked && feof(stream)
        error('BCP:TorchProtocol', 'Torch worker returned a truncated payload.');
    else
        fclear(stream);
        if toc(started) > timeout_seconds(), error('BCP:TorchTimeout', 'Torch payload timed out.'); end
        pause(0.001);
    end
end
value = typecast(raw, 'double');
[~, ~, endian] = computer();
if endian ~= 'L', value = swapbytes(value); end
value = value(:);
end

function close_worker(input, output, pid)
% EOF shuts down a healthy worker; bounded reaping also handles failed workers.
try, fclose(input); catch, end
try, fclose(output); catch, end
for attempt = 1:50
    [ended, ~] = waitpid(pid, WNOHANG());
    if ended ~= 0, return; end
    pause(0.01);
end
kill(pid, 15);
for attempt = 1:50
    [ended, ~] = waitpid(pid, WNOHANG());
    if ended ~= 0, return; end
    pause(0.01);
end
kill(pid, 9);
waitpid(pid);
end
