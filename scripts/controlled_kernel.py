"""Frozen-primitive diagnostic for the first kernel in a captured scaffold.

This never edits the production evaluator: temporary copies hold upstream
products/magnitudes fixed to distinguish arithmetic amplification from formulas.
"""
from __future__ import annotations
import inspect
import json
from pathlib import Path
import textwrap
import types

import numpy as np
import scipy.io as sio


def isolate(snapshot: Path, folder: Path, device: str, octave: Path, module: Path, timeout: float, resume: bool = False):
    import controlled_compare as driver
    root=driver.ROOT
    folder.mkdir(parents=True,exist_ok=resume)
    import hashlib
    overlay=folder/'overlay'
    controller=driver.instrument_controller((root/'GIF/GIF/MatlabScripts/iterationOfNewtonWithScaffold.m').read_text())
    controller=driver.replace_once(controller,'    [R,p]=chol(H1);',
        "    bcp_controlled_kernel_state();\n    [R,p]=chol(H1);")
    entry=driver.replace_once((root/'tests/controlled_controller_replay.m').read_text(),'iterationOfNewtonWithScaffold;',
        "try\n    iterationOfNewtonWithScaffold;\ncatch problem\n    if strcmp(problem.identifier,'BCP:ControlledKernelCaptured')\n        bcp_torch_energy('close');\n        return;\n    end\n    rethrow(problem);\nend\nerror('BCP:ControlledKernel','Expected kernel capture did not occur');")
    expected={'iterationOfNewtonWithScaffold.m':controller.encode(),'kernel_replay.m':entry.encode()}
    for name in ('bcp_controlled_capture_state.m','bcp_controlled_trace_state.m','bcp_controlled_kernel_state.m'):
        expected[name]=(root/'tests'/name).read_bytes()
    if not overlay.exists():
        driver.create_overlay(overlay,module)
        for name,data in expected.items():(overlay/name).write_bytes(data)
    if {path.name for path in overlay.iterdir()} != set(expected) | {module.name}:
        raise ValueError('Kernel overlay file inventory changed')
    overlay_hashes={}
    for name,data in expected.items():
        digest=hashlib.sha256(data).hexdigest()
        if driver.sha256(overlay/name)!=digest:raise ValueError('Kernel overlay/source provenance mismatch: '+name)
        overlay_hashes[name]=digest
    if driver.sha256(overlay/module.name)!=driver.sha256(module):raise ValueError('Kernel bridge provenance mismatch')
    overlay_hashes[module.name]=driver.sha256(module)
    reference_dir=folder/'reference'
    reference_file=reference_dir/'kernel.mat'
    env={'BCP_CONTROLLED_ROOT':str(root),'BCP_CONTROLLED_SNAPSHOT':str(snapshot),
         'BCP_CONTROLLED_RESULT':str(reference_dir/'unused.mat'),'BCP_CONTROLLED_OVERLAY':str(overlay),
         'BCP_CONTROLLED_EVALUATOR':'native','BCP_CONTROLLED_DEVICE':'cpu',
         'BCP_CONTROLLED_TRACE_DIR':'','BCP_CONTROLLED_SNAPSHOT_DIR':'',
         'BCP_CONTROLLED_KERNEL_REFERENCE':str(reference_file),'BCP_TORCH_NATIVE_DIR':str(overlay),
         'BCP_TORCH_MODULE_DIR':str(root/'BCP/GIF_and_CM')}
    driver.checked_native(octave,['--no-gui','--quiet',overlay/'kernel_replay.m'],env,reference_dir,timeout,resume)
    kernel=driver.load_struct(reference_file,'kernel');s,r,meta=kernel['s'],kernel['r'],kernel['meta']
    import sys
    sys.path.insert(0,str(root/'BCP/GIF_and_CM'))
    import torch
    import torch_energy as energy
    evaluator=energy.TorchEnergy(device)
    evaluator.prepare(s['J_fz'],s['J_fbz'],s['area'],s['fixed_indices'],s['fixed_values'],s['selection'])
    evaluator.forensic={key:torch.tensor(value,device=device,dtype=torch.complex128 if np.iscomplexobj(value) else torch.float64)
                       for key,value in r.items() if key.startswith(('raw_','grad_h_','hess_h'))}
    source=textwrap.dedent(inspect.getsource(energy.TorchEnergy.evaluate_device))
    source=driver.replace_once(source,'    # Lazy conjugate views avoid two extra persistent complex matrix copies.',
        "    self.forensic_raw = {'abs_fbz': abs_fbz, 'sigma': sigma, 'SIGMA': SIGMA}\n    # Lazy conjugate views avoid two extra persistent complex matrix copies.")
    selected=np.asarray(s['selection']).astype(int).reshape(-1)-1
    variants={};scalars={}
    for level,name in enumerate(('production','fixed_complex_products','fixed_complex_magnitudes','fixed_scalar_coefficients')):
        code=source
        if level>=1:
            code=driver.replace_once(code,'    fz = J_fz @ x_complex\n    fbz = J_fbz @ x_complex',
                "    fz = self.forensic['raw_fz']\n    fbz = self.forensic['raw_fbz']")
        if level>=2:
            code=driver.replace_once(code,'    abs_fz = fz.abs()\n    abs_fbz = fbz.abs()',
                "    abs_fz = self.forensic['raw_abs_fz']\n    abs_fbz = self.forensic['raw_abs_fbz']")
        if level>=3:
            for variable,expression in [('grad_h_SIGMA','area * ((SIGMA.pow(4) - 1) / SIGMA.pow(3))'),
                                       ('grad_h_sigma','area * ((sigma.pow(4) - 1) / sigma.pow(3))'),
                                       ('hess_h11','area * (1 + 3 * SIGMA.pow(-4))'),
                                       ('hess_h22','area * (1 + 3 * sigma.pow(-4))')]:
                code=driver.replace_once(code,f'    {variable} = {expression}',f"    {variable} = self.forensic['{variable}']")
        namespace=dict(vars(energy));exec(compile(code,'<controlled-kernel-'+name+'>','exec'),namespace)
        evaluator.evaluate_device=types.MethodType(namespace['evaluate_device'],evaluator)
        result=evaluator.evaluate(s['x'])
        references={'E':r['E'],'G1':np.asarray(r['G1']).reshape(-1),'H1':r['H1']}
        candidates={'E':result['E'],'G1':result['G'].reshape(-1)[selected],'H1':result['H']}
        variants[name]={key:driver.array_difference(references[key],np.asarray(candidates[key]).reshape(np.shape(references[key])),rtol=driver.KERNEL_RTOL,atol=0) for key in references}
        cpu_abs=np.asarray(r['raw_abs_fbz']).reshape(-1)
        gpu_abs=evaluator.forensic_raw['abs_fbz'].cpu().numpy().reshape(-1)
        sigma=evaluator.forensic_raw['sigma'].cpu().numpy().reshape(-1)
        cpu_zero=cpu_abs==0;gpu_zero=gpu_abs==0
        scalars[name]={'reference_exact_zero_fbz':int(cpu_zero.sum()),'candidate_exact_zero_fbz':int(gpu_zero.sum()),
            'different_zero_branch_rows':(np.flatnonzero(cpu_zero!=gpu_zero)+1).tolist(),
            'minimum_abs_fbz':float(gpu_abs.min()),'minimum_sigma':float(sigma.min())}
    evaluator._prepared=None
    code_agrees=all(item['passed'] for item in variants['fixed_complex_magnitudes'].values())
    report={'schema_version':1,'scope':'first evaluation of a captured scaffold, diagnostic upstream-value overrides only',
            'source_energy_sha256':driver.sha256(root/'BCP/GIF_and_CM/torch_energy.py'),
            'analysis_source_sha256':driver.sha256(Path(__file__)),'kernel_overlay_hashes':overlay_hashes,
            'resume_policy':'native reference stages are integrity-checked and reused; small frozen Torch variant analysis is recomputed',
            'reference_file_sha256':driver.sha256(reference_file),'snapshot_sha256':driver.sha256(snapshot),
            'device':device,'faces':int(np.asarray(s['J_fz']).shape[0]),'complex_dofs':int(np.asarray(s['J_fz']).shape[1]),
            'condition_H1':float(meta['condition_H1']) if np.isfinite(meta['condition_H1']) else None,
            'variants':variants,'scalar_branches':scalars,
            'analytic_formulas_agree_when_upstream_products_and_magnitudes_fixed':code_agrees,
            'production_kernel_envelope_passed':all(v['passed'] for v in variants['production'].values()),
            'historical_parity_certified':False}
    recorded=snapshot.parent.parent/'controller_steps'/(snapshot.stem+'_step_0001.mat')
    if recorded.is_file():
        trace=driver.load_struct(recorded,'state')
        report['reference_matches_recorded_native_kernel']={key:driver.array_difference(trace[key],r[key],rtol=0,atol=0)
            for key in ('E','G1','H1')}
    report['classification']=('exact-zero-fbz-branch-sensitive' if scalars['production']['different_zero_branch_rows'] and code_agrees
        else ('upstream-products-or-magnitudes-amplified' if code_agrees else 'additional-formula-or-reduction-difference-unresolved'))
    (folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report
