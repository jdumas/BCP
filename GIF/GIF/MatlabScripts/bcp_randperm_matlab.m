function p = bcp_randperm_matlab(n, seed)
% Preserve main's MATLAB rng(seed); randperm(n) sampling in Octave.
% MATLAB's full, one-argument randperm sorts uniform MT19937 draws; Octave
% uses a different seed initializer and a shuffle. Both differences change
% which interior faces enter GIF's objective. This is not a general RNG shim.
% References: MathWorks randperm documentation and MATLAB Answers 12248
% (full randperm retains sort(rand(1,n)) for backwards compatibility).
if nargin < 2, seed = 1; end
if ~isscalar(n) || ~isreal(n) || ~isfinite(n) || n < 0 || n ~= fix(n)
    error('BCP:Sampling', 'Permutation size must be a nonnegative integer.');
end
if ~isscalar(seed) || ~isreal(seed) || ~isfinite(seed) || seed < 0 || seed > 2^32-1 || seed ~= fix(seed)
    error('BCP:Sampling', 'Seed must be an unsigned 32-bit integer.');
end
if exist('OCTAVE_VERSION', 'builtin') == 0
    rng(seed);
    p = randperm(n);
    return;
end
% MATLAB's factory seed 0 denotes the reference MT19937 default seed 5489.
if seed == 0, seed = 5489; end
state = zeros(625, 1, 'uint32');
state(1) = uint32(seed);
for k = 1:623
    value = uint64(1812433253) * uint64(bitxor(state(k), bitshift(state(k), -30))) + uint64(k);
    state(k+1) = uint32(bitand(value, uint64(intmax('uint32'))));
end
state(625) = uint32(1);
rand('state', state);
[~, p] = sort(rand(1, n));
end
