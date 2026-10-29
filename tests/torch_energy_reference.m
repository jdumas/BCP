% Compare all six outputs through the real persistent pipe to the original .m.
1;
function close_worker_bridge()
    bcp_torch_energy('close');
end
function check_close(actual, expected, label)
    assert(isequal(size(actual), size(expected)), ['Shape mismatch: ', label]);
    if isempty(expected), return; end
    error_size = max(abs(actual(:) - expected(:)));
    scale = max(1, max(abs(expected(:))));
    assert(error_size <= 2e-9 * scale, sprintf('%s mismatch: %g / %g', label, error_size, scale));
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'GIF', 'GIF', 'MatlabScripts'));
device = getenv('BCP_TORCH_REFERENCE_DEVICE');
if isempty(device), device = 'cpu'; end
info = bcp_torch_energy('start', device);
cleanup = onCleanup(@close_worker_bridge);
assert(strcmp(info.device, device));
pid = info.pid;
rng(42);
N = 4;
F = 9;
J = randn(F,N) + 1i * randn(F,N);
x_full = randn(2*N,1);
area = 0.1 + rand(F,1);
evaluations = 0;
for variant = 1:5
    B = bsxfun(@times, J, 0.15 * exp(1i * rand(F,1)));
    if variant == 2, B(1:3,:) = 10 * B(1:3,:); end
    if variant == 3, B = zeros(F,N); end
    if variant == 4, B = 2 * J; end
    current_x = x_full;
    if variant == 5, current_x = 0.1 * x_full; end
    for fixed_case = 1:3
        fixed = [];
        if fixed_case == 2, fixed = [1; N+1]; end
        if fixed_case == 3, fixed = (1:2*N)'; end
        free = setdiff((1:2*N)', fixed);
        bcp_torch_energy('prepare', J, B, area, fixed, current_x(fixed));
        before = bcp_torch_energy('stats');
        for spd = [false true]
            for include_flips = [0 1]
                [fi0,fz0,fbz0,E0,G0,H0] = symmetricDirichletEnergy(current_x(free),J,B,area,fixed,current_x(fixed),spd,include_flips);
                [fi,fz,fbz,E,G,H] = bcp_torch_energy('evaluate',current_x(free),spd,include_flips);
                check_close(fi,fi0,'foldovers');
                check_close(fz,fz0,'fz'); check_close(fbz,fbz0,'fbz');
                check_close(E,E0,'energy'); check_close(G,G0,'gradient'); check_close(H,H0,'Hessian');
                evaluations = evaluations + 1;
            end
        end
        after = bcp_torch_energy('stats');
        assert(after.pid == pid && after.prepare_count == before.prepare_count);
        assert(after.matrix_upload_bytes == before.matrix_upload_bytes);
    end
end
% A nonzero fbz Jacobian that annihilates x exercises the nontrivial zero-fbz convention.
xc = [1+2i; 3-1i];
x = [real(xc); imag(xc)];
J = [2-1i, 0.5+1i]; B = [xc(2), -xc(1)];
bcp_torch_energy('prepare', J, B, 1.25, [], []);
for spd = [false true]
    [fi0,fz0,fbz0,E0,G0,H0] = symmetricDirichletEnergy(x,J,B,1.25,[],[],spd,0);
    [fi,fz,fbz,E,G,H] = bcp_torch_energy('evaluate',x,spd,0);
    assert(fbz == 0);
    check_close(fi(:),fi0(:),'zero-fbz foldovers'); check_close(fz,fz0,'zero-fbz fz');
    check_close(fbz,fbz0,'zero-fbz fbz'); check_close(E,E0,'zero-fbz E');
    check_close(G,G0,'zero-fbz G'); check_close(H,H0,'zero-fbz H');
    evaluations = evaluations + 1;
end
% Exercise payloads larger than the pipe buffer and reuse a prepared Jacobian.
F = 129; N = 65;
J = randn(F,N) + 1i * randn(F,N);
B = 0.1 * J;
x = randn(2*N,1);
area = ones(F,1)/F;
bcp_torch_energy('prepare',J,B,area,[],[]);
before = bcp_torch_energy('stats');
for step = 1:2
    [fi0,fz0,fbz0,E0,G0,H0] = symmetricDirichletEnergy(x,J,B,area,[],[],true,0);
    [fi,fz,fbz,E,G,H] = bcp_torch_energy('evaluate',x,true,0);
    check_close(fi,fi0,'large foldovers'); check_close(fz,fz0,'large fz');
    check_close(fbz,fbz0,'large fbz'); check_close(E,E0,'large E');
    check_close(G,G0,'large G'); check_close(H,H0,'large H');
    x = x + 0.01;
    evaluations = evaluations + 1;
end
after = bcp_torch_energy('stats');
assert(after.pid == pid && after.prepare_count == before.prepare_count);
assert(after.matrix_upload_bytes == before.matrix_upload_bytes);
selection = [2*N; 1; 3];
bcp_torch_energy('prepare', J, B, area, [], [], selection);
[~,~,~,~,~,H0] = symmetricDirichletEnergy(x,J,B,area,[],[],true,0);
[~,~,~,~,~,H] = bcp_torch_energy('evaluate',x,true,0);
check_close(H,H0(selection,selection),'selected Hessian block');
evaluations = evaluations + 1;
after = bcp_torch_energy('stats');
assert(after.evaluate_count == evaluations);
% A numerical request failure must not leave a usable stale workspace/cache.
caught = false;
try, bcp_torch_energy('evaluate', 0, true, 0); catch, caught = true; end
assert(caught);
caught = false;
try, bcp_torch_energy('stats'); catch, caught = true; end
assert(caught);
info = bcp_torch_energy('start', 'cpu');
assert(info.prepare_count == 0 && info.evaluate_count == 0);
bcp_torch_energy('close');
fprintf('Torch evaluator (%s) matches original Octave E/G/H in %d evaluations; persistent cache/no per-step uploads and error recovery verified.\n', device, evaluations);
