function bcp_controlled_trace_state(stage, reason)
% Diagnostic I/O only; caller equations, branches and floating-point values stay unchanged.
persistent state;
if strcmp(stage,'reset'), state=[]; return; end
folder=getenv('BCP_CONTROLLED_TRACE_DIR');
if isempty(folder), return; end
if ~exist(folder,'dir'), mkdir(folder); end
if nargin<2, reason=''; end
if strcmp(stage,'chol_primary')
    state=struct('schema_version',1,'primary_chol_p',evalin('caller','p'),'fallback',0);
    for key={'E','Eprev','G','G1','H1','fz','fbz','foldoversIndices','x_initial','timesRemeshed','i','iter'}
        name=key{1};state.(name)=evalin('caller',name);
    end
    state.x_before=state.x_initial;state=rmfield(state,'x_initial');
elseif strcmp(stage,'chol_shifted') || strcmp(stage,'chol_identity')
    if isempty(state), error('BCP:ControlledTrace','Missing primary Cholesky state'); end
    state.fallback=1+strcmp(stage,'chol_identity');state.final_chol_p=evalin('caller','p');
elseif strcmp(stage,'step')
    if isempty(state), error('BCP:ControlledTrace','Missing evaluation state'); end
    for key={'d','d1','t','tMax','stepTol','gradTol','Etol','iter'}
        name=key{1};state.(name)=evalin('caller',name);
    end
    state.x_after=evalin('caller','x_initial');
    path=fullfile(folder,sprintf('remesh_%04d_step_%04d.mat',state.timesRemeshed,state.i));
    if exist(path,'file'), error('BCP:ControlledTrace',['Refusing to overwrite ',path]); end
    save('-mat-binary',path,'state');state=[];
elseif strcmp(stage,'stop')
    terminal=struct('schema_version',1,'reason',reason,'remesh',evalin('caller','timesRemeshed'), ...
                    'step',evalin('caller','i'),'iterations',evalin('caller','iter'));
    path=fullfile(folder,sprintf('remesh_%04d_stop.mat',terminal.remesh));
    if exist(path,'file'),error('BCP:ControlledTrace',['Refusing to overwrite ',path]);end
    save('-mat-binary',path,'terminal');
else
    error('BCP:ControlledTrace',['Unknown trace stage ',stage]);
end
end
