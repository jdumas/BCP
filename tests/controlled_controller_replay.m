% Replay one captured original controller state. No synthetic tolerances or re-initialization.
1;
root=getenv('BCP_CONTROLLED_ROOT');input=getenv('BCP_CONTROLLED_SNAPSHOT');
output=getenv('BCP_CONTROLLED_RESULT');overlay=getenv('BCP_CONTROLLED_OVERLAY');
if isempty(root) || isempty(input) || isempty(output) || isempty(overlay)
    error('BCP:ControlledReplay','Missing root/snapshot/result/overlay configuration');
end
addpath(fullfile(root,'GIF','GIF','MatlabScripts'));
native=getenv('BCP_TORCH_NATIVE_DIR');if ~isempty(native),addpath(native);end
addpath(overlay);
loaded=load(input);snapshot=loaded.snapshot;
if snapshot.schema_version~=1,error('BCP:ControlledReplay','Unsupported state schema');end
GIF=snapshot.gif;
for key={'SF','SV','SJ_fz','SJ_fbz','SArea','EditedJ_fz','EditedJ_fbz'}
    GIF.(key{1})=[];
end
allowed={'x_initial','numDOF','numDOF1','numDOFScaffold','totVars','almostAllVars', ...
 'numR','iter','timesRemeshed','stepTol1','gradTol1','Etol1','EitersCounter','numConvIters', ...
 'rate','lambda','areaOriginal','terminationCondition','numFixedDOFScaffold'};
for key=fieldnames(snapshot.vars)'
    name=key{1};if ~any(strcmp(name,allowed)),error('BCP:ControlledReplay','Unexpected state variable');end
    eval([name,'=snapshot.vars.(name);']);
end
assert(totVars==numel(x_initial));assert(numDOFScaffold==columns(GIF.TJ_fz));
assert(numDOF1+numDOF==numDOFScaffold);
assert(2*numDOFScaffold==totVars+numel(GIF.FixedIndices));
mode=getenv('BCP_CONTROLLED_EVALUATOR');device=getenv('BCP_CONTROLLED_DEVICE');
if strcmp(mode,'torch')
    if isempty(device),error('BCP:ControlledReplay','Explicit Torch device required');end
    info=bcp_torch_energy('start',device);
    assert(strcmp(info.device,device));assert(strcmp(info.backend,'inprocess'));
    GIF.useGPU=strncmp(device,'cuda',4);GIF.computeBackend='torch';
elseif strcmp(mode,'native')
    GIF.useGPU=false;GIF.computeBackend='native';
else
    error('BCP:ControlledReplay','Evaluator must be native or torch');
end
bcp_controlled_trace_state('reset');
iterationOfNewtonWithScaffold;
result=struct('schema_version',1,'coordinates',x_initial,'cones',GIF.UVonCones, ...
 'energy',GIF.NewtonFinalEnergy,'iterations',GIF.newtonIters,'remesh',timesRemeshed);
save('-mat-binary',getenv('BCP_CONTROLLED_RESULT'),'result');
bcp_torch_energy('close');
