% Original deterministic interior-face sampling must not change in Octave.
1;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'GIF', 'GIF', 'MatlabScripts'));
% Published MathWorks randperm examples from factory RNG defaults.
assert(isequal(bcp_randperm_matlab(6,0), [6 3 5 1 2 4]));
assert(isequal(bcp_randperm_matlab(10,0), [6 3 7 8 5 1 2 4 9 10]));
% main explicitly uses seed 1. Independent reference MT19937/NumPy RandomState
% draws sorted stably give these indices, including across many MT twists.
assert(isequal(bcp_randperm_matlab(10,1), [3 6 5 7 4 8 9 1 10 2]));
p = bcp_randperm_matlab(100000,1);
assert(isequal(p(1:20), [47309 48320 94377 67428 55031 11064 46064 61799 31415 79319 11847 41222 48184 45681 3600 3 67031 59905 56431 4971]));
assert(isequal(sort(p), 1:100000));
assert(isequal(bcp_randperm_matlab(100000,1),p));
assert(isequal(bcp_randperm_matlab(0,1),zeros(1,0)));
assert(isequal(bcp_randperm_matlab(1,1),1));
for input = {-1,1.5,NaN,Inf,[1 2],1i}
    rejected=false;
    try, bcp_randperm_matlab(input{1},1); catch, rejected=true; end
    assert(rejected);
end
for input = {-1,1.5,NaN,Inf,2^32,[1 2],1i}
    rejected=false;
    try, bcp_randperm_matlab(10,input{1}); catch, rejected=true; end
    assert(rejected);
end
fprintf('MATLAB-compatible interior sampling fixtures and repeatability passed\n');
