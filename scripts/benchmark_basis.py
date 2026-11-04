#!/usr/bin/env python3
"""Isolated oneMKL solve probe on an existing KKT dump; never changes BCP controls.

Tests documented partial-output/sparse-RHS modes and batching against previously
validated full inverse-column solves, including switching back to a full solve.
No Schur reformulation, dense inverse, precision change or silent fallback.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import check_kkt

MODES={'full':0,'sparse-selected':1,'sparse-rhs':2,'selected':3}


class Pardiso:
    def __init__(self,library:Path,A,mode:int,requested,rhs_rows):
        if os.environ.get('MKL_INTERFACE_LAYER','LP64')!='LP64':
            raise ValueError('This probe requires the production LP64 interface')
        self.lib=ct.CDLL(str(library));self.pt=(ct.c_void_p*64)();self.ip=np.zeros(64,np.int32)
        self.n=ct.c_int(A.shape[0]);self.kind=ct.c_int(-2);self.maxfct=ct.c_int(1);self.mnum=ct.c_int(1);self.msg=ct.c_int(0)
        if A.shape[0]>=np.iinfo(np.int32).max or A.nnz>=np.iinfo(np.int32).max:raise ValueError('Matrix exceeds LP64 index range')
        U=sp.triu(A,format='csr');U.sort_indices()
        self.values=np.ascontiguousarray(U.data,np.float64);self.ia=np.ascontiguousarray(U.indptr+1,np.int32);self.ja=np.ascontiguousarray(U.indices+1,np.int32)
        self.perm=np.zeros(A.shape[0],np.int32)
        pi=ct.POINTER(ct.c_int);pd=ct.c_void_p;pp=ct.POINTER(ct.c_void_p)
        self.lib.pardisoinit.argtypes=[pp,pi,pi];self.lib.pardisoinit.restype=None
        self.lib.pardiso.argtypes=[pp,pi,pi,pi,pi,pi,pd,pi,pi,pi,pi,pi,pi,pd,pd,pi];self.lib.pardiso.restype=None
        self.lib.pardisoinit(self.pt,ct.byref(self.kind),self.ip.ctypes.data_as(pi))
        self.ip[0]=1;self.ip[1]=3;self.ip[7]=0;self.ip[10]=1;self.ip[12]=1
        self.ip[17]=-1;self.ip[27]=0;self.ip[30]=mode;self.ip[34]=0;self.ip[35]=0;self.ip[59]=0
        # All supplied to every phase; MKL overwrites this with its internal permutation.
        if mode==1:self.perm[np.union1d(requested,rhs_rows)]=1
        elif mode==2:self.perm[rhs_rows]=1
        elif mode==3:self.perm[requested]=1
        self.started=False
        version=ct.create_string_buffer(512);self.lib.MKL_Get_Version_String(version,ct.c_int(512));self.version=version.value.decode()
        self.lib.MKL_Get_Max_Threads.restype=ct.c_int;self.threads=self.lib.MKL_Get_Max_Threads()

    def call(self,phase:int,B=None,X=None):
        if B is None:B=np.zeros((1,1),np.float64,order='F')
        if X is None:X=np.zeros_like(B,order='F')
        p=ct.c_int(phase);nrhs=ct.c_int(B.shape[1]);error=ct.c_int(0);pi=ct.POINTER(ct.c_int)
        self.started=True
        start=time.perf_counter()
        self.lib.pardiso(self.pt,ct.byref(self.maxfct),ct.byref(self.mnum),ct.byref(self.kind),ct.byref(p),ct.byref(self.n),
            self.values.ctypes.data,self.ia.ctypes.data_as(pi),self.ja.ctypes.data_as(pi),self.perm.ctypes.data_as(pi),
            ct.byref(nrhs),self.ip.ctypes.data_as(pi),ct.byref(self.msg),B.ctypes.data,X.ctypes.data,ct.byref(error))
        elapsed=time.perf_counter()-start
        if error.value:raise RuntimeError(f'MKL PARDISO phase{phase} returned error{error.value}')
        return elapsed

    def close(self):
        if self.started:
            self.call(-1);self.started=False


def load_problem(directory:Path|None):
    if directory is None:
        A=sp.csr_matrix([[4.,1.,1.],[1.,-2.,0.],[1.,0.,3.]])
        cols=np.array([0,1,2],dtype=int);rows=np.array([0,2],dtype=int)
        ref=np.linalg.solve(A.toarray(),np.eye(3))
        return A,rows,cols,ref[rows],ref
    passes=check_kkt.pass_directories(directory)
    if len(passes)!=1:raise ValueError('Select exactly one KKT pass')
    p=passes[0];A=check_kkt.read_matrix(p);meta=check_kkt.read_metadata(p,len(A.indptr)-1)
    if meta['kind']!='kkt':raise ValueError('This probe needs a symmetric KKT pass')
    rows=np.asarray(meta['selected_rows'],dtype=int);cols=np.asarray(meta['selected_columns'],dtype=int)
    if not len(rows) or not len(cols):raise ValueError('Selected row/column metadata is required')
    ref=np.empty((len(rows),len(cols)));position={int(c):i for i,c in enumerate(cols)}
    # Reference full-column dumps were independently validated by check_kkt.py.
    full_back=np.zeros((A.shape[0],2));weights=np.column_stack((np.sin(np.arange(len(cols))+.25),np.cos(np.arange(len(cols))+.5)))
    coverage=[]
    for path in sorted(p.glob('batch_*.mat')):
        data=sio.loadmat(path);X=data['X'];which=np.asarray(data['columns']).astype(int).reshape(-1)
        positions=[position[int(c)] for c in which];ref[:,positions]=X[rows];full_back+=X@weights[positions];coverage.extend(which.tolist())
    if sorted(coverage)!=sorted(cols.tolist()):raise ValueError('Reference columns do not match metadata')
    return A,rows,cols,ref,full_back


def probe(args):
    A,rows,cols,reference,full_reference=load_problem(args.dump)
    n=A.shape[0];batch=min(args.batch_size,len(cols));mode=MODES[args.mode]
    result={'schema_version':1,'measurement_scope':'isolated_vendor_solver_probe_not_whole_BCP',
        'mode':args.mode,'mtype':-2,'n':n,'nnz':A.nnz,'requested_rows':len(rows),'rhs_columns':len(cols),
        'batch_size':batch,'rounds':args.rounds,'library':str(args.mkl_library),
        'environment':{k:os.environ.get(k) for k in ('MKL_NUM_THREADS','OMP_NUM_THREADS','MKL_THREADING_LAYER','MKL_INTERFACE_LAYER')},
        'dense_rhs_and_solution_buffer_bytes':int(2*n*batch*8),'production_backend_modified':False,'ok':False}
    solver=None
    try:
        solver=Pardiso(args.mkl_library,A,mode,rows,cols)
        result.update(mkl_version=solver.version,effective_mkl_threads=solver.threads)
        result['analysis_and_factorization_seconds']=solver.call(12)
        result['factor_nonzeros']=int(solver.ip[17]);result['perturbed_pivots']=int(solver.ip[13])
        B=np.zeros((n,batch),np.float64,order='F');X=np.empty_like(B,order='F')
        times=[];maximum_error=0.;maximum_scaled=0.;round_loop=[]
        scale=max(float(np.max(abs(reference))),np.finfo(float).tiny)
        for repeat in range(args.rounds+1):
            elapsed=0.;loop_start=time.perf_counter()
            for first in range(0,len(cols),batch):
                count=min(batch,len(cols)-first)
                b=B[:,:count];x=X[:,:count];b.fill(0);x.fill(np.nan)
                b[cols[first:first+count],np.arange(count)]=1
                elapsed+=solver.call(33,b,x)
                current=x[rows];expected=reference[:,first:first+count]
                if not np.isfinite(current).all():raise RuntimeError('Requested output contains nonfinite values')
                error=float(np.max(abs(current-expected),initial=0));maximum_error=max(maximum_error,error);maximum_scaled=max(maximum_scaled,error/scale)
            if repeat:
                times.append(elapsed);round_loop.append(time.perf_counter()-loop_start)
        result.update(phase33_seconds=times,phase33_median_seconds=float(np.median(times)),
            loop_with_validation_seconds=round_loop,selected_entries_max_abs_difference=maximum_error,
            selected_entries_relative_difference=maximum_scaled)
        # Original BCP later reuses these factors for a full UV back-substitution.
        # A partial mode that cannot safely do this is not usable, however fast.
        solver.ip[30]=0
        weights=np.column_stack((np.sin(np.arange(len(cols))+.25),np.cos(np.arange(len(cols))+.5)))
        rhs=np.zeros((n,2),np.float64,order='F');rhs[cols]=weights
        actual=np.empty_like(rhs,order='F');result['full_solve_after_partial_seconds']=solver.call(33,rhs,actual)
        if not np.isfinite(actual).all():raise RuntimeError('Full factor-reuse solve contains nonfinite values')
        if args.dump is None:full_reference=full_reference@weights
        residual=np.linalg.norm(A@actual-rhs);anorm=sp.linalg.norm(A);xnorm=np.linalg.norm(actual);bnorm=np.linalg.norm(rhs)
        result['full_solve_after_partial_backward_error']=check_kkt.normalized_backward_error(residual,anorm,xnorm,bnorm)
        result['full_solve_after_partial_relative_difference']=check_kkt.relative_difference(actual,full_reference)
        result['ok']=bool(maximum_scaled<=check_kkt.DEFAULT_SOLUTION_TOLERANCE and
            result['full_solve_after_partial_backward_error']<=check_kkt.DEFAULT_BACKWARD_TOLERANCE and
            result['full_solve_after_partial_relative_difference']<=check_kkt.DEFAULT_SOLUTION_TOLERANCE)
    except (OSError,ValueError,RuntimeError) as error:result['error']=str(error)
    finally:
        if solver is not None:
            try:solver.close()
            except RuntimeError as error:result['cleanup_error']=str(error);result['ok']=False
    result['process_peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump',type=Path,help='One existing independently checked KKT dump; omit for tiny contract fixture')
    parser.add_argument('--mode',choices=MODES,default='full')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mkl-library',type=Path,default=Path(sys.prefix)/'lib/libmkl_rt.so.3')
    args=parser.parse_args(argv)
    if args.batch_size<1 or args.rounds<1:parser.error('batch size and rounds must be positive')
    result=probe(args);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False));return 0 if result['ok'] else 2


if __name__=='__main__':raise SystemExit(main())
