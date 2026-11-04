function bcp_controlled_kernel_state()
% Called only from the kernel-isolation overlay, before the original chol call.
path=getenv('BCP_CONTROLLED_KERNEL_REFERENCE');
if isempty(path),return;end
GIF=evalin('caller','GIF');
if ~strcmp(GIF.computeBackend,'native'),error('BCP:ControlledKernel','CPU .m reference required');end
s=struct('J_fz',GIF.TJ_fz,'J_fbz',GIF.TJ_fbz,'area',GIF.TArea, ...
 'fixed_indices',GIF.FixedIndices,'fixed_values',GIF.FixedValues,'x',evalin('caller','x_initial'));
n=columns(s.J_fz);full=zeros(2*n,1);free=setdiff((1:(2*n))',s.fixed_indices);
full(free)=s.x;full(s.fixed_indices)=s.fixed_values;z=complex(full(1:n),full(n+1:end));
r=struct();r.raw_fz=s.J_fz*z;r.raw_fbz=s.J_fbz*z;
folds=evalin('caller','foldoversIndices');keep=true(rows(s.J_fz),1);keep(folds)=false;
% Preserve the EXACT products returned by the source evaluator for retained rows.
r.raw_fz(keep)=evalin('caller','fz');r.raw_fbz(keep)=evalin('caller','fbz');
r.raw_abs_fz=abs(r.raw_fz);r.raw_abs_fbz=abs(r.raw_fbz);
r.raw_SIGMA=r.raw_abs_fz+r.raw_abs_fbz;r.raw_sigma=r.raw_abs_fz-r.raw_abs_fbz;
r.foldovers=folds;r.E=evalin('caller','E');r.G1=evalin('caller','G1');r.H1=evalin('caller','H1');
A=s.area(keep);large=r.raw_SIGMA(keep);small=r.raw_sigma(keep);
r.grad_h_SIGMA=A.*((large.^4-1)./large.^3);r.grad_h_sigma=A.*((small.^4-1)./small.^3);
r.hess_h11=A.*(1+3*large.^(-4));r.hess_h22=A.*(1+3*small.^(-4));
r.beta=r.grad_h_SIGMA-r.grad_h_sigma;r.beta_multiplier=r.beta./r.raw_abs_fbz(keep);
r.beta_multiplier(r.raw_abs_fbz(keep)==0)=1;
outer=evalin('caller','numDOF1');ndof=evalin('caller','numDOF');total=evalin('caller','totVars');almost=evalin('caller','almostAllVars');
s.selection=[outer:(n-1),(almost+1):total]';
meta=struct('n',n,'outer',outer,'dofs',ndof,'condition_H1',cond(r.H1),'exact_zero_fbz',sum(r.raw_abs_fbz==0), ...
 'min_retained_sigma',min(small),'largest_cancellation_ratio',max(large./small));
kernel=struct('schema_version',1,'s',s,'r',r,'meta',meta);
save('-mat-binary',path,'kernel');
error('BCP:ControlledKernelCaptured','Frozen kernel reference captured');
end
