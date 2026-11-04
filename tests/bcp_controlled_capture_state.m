function bcp_controlled_capture_state()
% Validation-only hook called by an instrumented COPY of the original script.
folder = getenv('BCP_CONTROLLED_SNAPSHOT_DIR');
if isempty(folder), return; end
if ~exist(folder,'dir'), mkdir(folder); end
GIF = evalin('caller','GIF');
snapshot = struct('schema_version',1,'gif',struct(),'vars',struct());
for key = {'TJ_fz','TJ_fbz','TArea','FixedIndices','FixedValues','ScaffoldUVonCones'}
    name=key{1};
    if ~isfield(GIF,name), error('BCP:ControlledState',['Missing GIF.',name]); end
    snapshot.gif.(name)=GIF.(name);
end
names={'x_initial','numDOF','numDOF1','numDOFScaffold','totVars','almostAllVars', ...
       'numR','iter','timesRemeshed','stepTol1','gradTol1','Etol1', ...
       'EitersCounter','numConvIters','rate','lambda','areaOriginal', ...
       'terminationCondition','numFixedDOFScaffold'};
for key=names
    name=key{1};
    if evalin('caller',['exist(''',name,''',''var'')'])
        snapshot.vars.(name)=evalin('caller',name);
    end
end
required={'x_initial','numDOF','numDOF1','numDOFScaffold','totVars','almostAllVars','iter','timesRemeshed'};
for key=required
    if ~isfield(snapshot.vars,key{1}), error('BCP:ControlledState',['Missing ',key{1}]); end
end
path=fullfile(folder,sprintf('remesh_%04d.mat',snapshot.vars.timesRemeshed+1));
if exist(path,'file'), error('BCP:ControlledState',['Refusing to overwrite ',path]); end
save('-mat-binary',path,'snapshot');
end
