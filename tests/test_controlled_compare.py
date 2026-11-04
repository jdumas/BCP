"""Controlled validation adds observability, never changes original equations/gates."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import subprocess
import types
import numpy as np
import scipy.io as sio

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('controlled_compare',ROOT/'scripts/controlled_compare.py')
controlled=importlib.util.module_from_spec(spec);spec.loader.exec_module(controlled)


class ControlledCompareTests(unittest.TestCase):
    def test_instrumentation_only_inserts_hook_calls(self):
        source=(ROOT/'GIF/GIF/MatlabScripts/iterationOfNewtonWithScaffold.m').read_text()
        instrumented=controlled.instrument_controller(source)
        restored=''.join(line for line in instrumented.splitlines(keepends=True)
                         if not line.strip().startswith(('bcp_controlled_capture_state(', 'bcp_controlled_trace_state(')))
        self.assertEqual(restored,source)
        self.assertIn("bcp_controlled_trace_state('chol_identity')",instrumented)
        self.assertIn("bcp_controlled_trace_state('stop','energy_counter')",instrumented)

    def test_instrumentation_fails_loudly_on_changed_source(self):
        with self.assertRaises(ValueError):controlled.instrument_controller('different controller')
        with self.assertRaises(ValueError):controlled.replace_once('x x','x','y')

    def test_nonfinite_metric_is_retained_as_numerical_failure(self):
        report={'ok':True,'primary_checks_passed':True,'pipeline_metric':float('inf')}
        encoded=json.loads(controlled.encode_report(report))
        self.assertIsNone(encoded['pipeline_metric'])
        self.assertFalse(encoded['ok']);self.assertFalse(encoded['primary_checks_passed'])
        self.assertIn('report.pipeline_metric',encoded['nonfinite_numerical_values'])
        self.assertNotIn('execution_error',encoded)

    def test_kernel_threshold_not_weakened(self):
        self.assertEqual(controlled.KERNEL_RTOL,2e-9)
        result=controlled.array_difference(np.array([1.]),np.array([1.+3e-9]),rtol=controlled.KERNEL_RTOL,atol=0)
        self.assertFalse(result['passed'])
        self.assertTrue(controlled.array_difference([1.],[1.+1e-10],rtol=2e-9,atol=0)['passed'])

    def test_nonfinite_and_wrong_shapes_fail(self):
        self.assertFalse(controlled.array_difference([1.],[float('nan')],rtol=1,atol=1)['passed'])
        self.assertFalse(controlled.array_difference(np.zeros((1,2)),np.zeros((2,1)),rtol=1,atol=1)['passed'])

    def test_input_hashes_detect_changes_and_unsafe_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);p=folder/'mesh.obj';p.write_text('data')
            manifest={'files':{'mesh.obj':{'sha256':controlled.sha256(p)}}}
            controlled.verify_inputs(folder,manifest)
            p.write_text('changed')
            with self.assertRaises(ValueError):controlled.verify_inputs(folder,manifest)
            with self.assertRaises(ValueError):controlled.verify_inputs(folder,{'files':{'../mesh.obj':{'sha256':'bad'}}})

    def test_freeze_inputs_copies_not_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir();(source/'frozen-source.obj').write_text('v 0 0 0\n')
            sio.savemat(source/'frozen-edge-lengths.mat',{'edge_lengths':np.ones((2,3))})
            target=Path(tmp)/'target';manifest=controlled.freeze_inputs(source,target)
            self.assertEqual(manifest['face_count'],2)
            (source/'frozen-source.obj').write_text('changed')
            self.assertEqual((target/'frozen-source.obj').read_text(),'v 0 0 0\n')
            controlled.verify_inputs(target,manifest)

    def test_invalid_frozen_lengths_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir();(source/'frozen-source.obj').write_text('data')
            sio.savemat(source/'frozen-edge-lengths.mat',{'edge_lengths':np.array([[1.,-1.,2.]])})
            with self.assertRaises(ValueError):controlled.freeze_inputs(source,Path(tmp)/'target')

    def test_artifact_symlink_cannot_escape_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'capture';folder.mkdir();outside=root/'outside';outside.write_text('private')
            (folder/'linked').symlink_to(outside)
            with self.assertRaises(ValueError):controlled.contained_file(folder,'linked')
            with self.assertRaises(ValueError):controlled.contained_file(folder,'../outside')

    def test_resume_reuses_only_verified_completed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'fake-executable';binary.write_text('not executed');logs=root/'logs';logs.mkdir()
            (logs/'stdout.log').write_text('recorded');(logs/'stderr.log').write_text('')
            marker={'configuration':{'executable_sha256':controlled.sha256(binary),'arguments':[],
                'environment':{},'runtime':controlled.runtime_fingerprint()},
                'files':{'stdout.log':controlled.sha256(logs/'stdout.log'),'stderr.log':controlled.sha256(logs/'stderr.log')}}
            (logs/'run-complete.json').write_text(json.dumps(marker))
            result=controlled.checked_native(binary,[],{},logs,1,resume=True)
            self.assertEqual(result.stdout,'recorded')
            extra=logs/'unexpected.mat';extra.write_text('unlisted')
            with self.assertRaises(ValueError):controlled.checked_native(binary,[],{},logs,1,resume=True)
            extra.unlink()
            with mock.patch.dict(controlled.os.environ,{'MKL_NUM_THREADS':'changed'}):
                with self.assertRaises(ValueError):controlled.checked_native(binary,[],{},logs,1,resume=True)
            (logs/'stdout.log').write_text('tampered')
            with self.assertRaises(ValueError):controlled.checked_native(binary,[],{},logs,1,resume=True)

    def test_resume_refuses_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'fake';binary.write_text('not executed');logs=root/'logs';logs.mkdir()
            (logs/'partial.mat').write_text('incomplete')
            with self.assertRaises(ValueError):controlled.checked_native(binary,[],{},logs,1,resume=True)
            self.assertEqual((logs/'partial.mat').read_text(),'incomplete')

    def test_snapshot_rejects_integer_index_abi_and_wrong_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'snapshot.mat'
            gif={'TJ_fz':np.ones((2,2),complex),'TJ_fbz':np.ones((2,2),complex),'TArea':np.ones((2,1)),
                 'FixedIndices':np.array([[1.],[3.]]),'FixedValues':np.ones((2,1)),'ScaffoldUVonCones':np.ones((2,1))}
            variables={k:float(v) for k,v in {'numDOFScaffold':2,'numDOF':1,'numDOF1':1,'totVars':2,'almostAllVars':1,'iter':0,'timesRemeshed':0}.items()}
            variables['x_initial']=np.ones((2,1))
            def save():sio.savemat(p,{'snapshot':{'schema_version':1,'gif':gif,'vars':variables}})
            save();controlled.validate_snapshot(p)
            variables['numDOF']=np.int64(1);save()
            with self.assertRaises(ValueError):controlled.validate_snapshot(p)
            variables['numDOF']=1.;gif['TJ_fz']=np.ones((2,2),dtype=np.float32);save()
            with self.assertRaises(ValueError):controlled.validate_snapshot(p)

    def test_kernel_overlay_tampering_is_rejected_before_execution(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        import controlled_kernel
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);module=folder/'bcp_torch_energy_native.oct';module.write_bytes(b'not executed')
            target=folder/'kernel';target.mkdir();controlled.create_overlay(target/'overlay',module)
            with self.assertRaisesRegex(ValueError,'provenance mismatch|inventory changed'):
                controlled_kernel.isolate(folder/'snapshot.mat',target,'cpu',folder/'octave',module,1,resume=True)

    def test_diagnostic_failure_preserves_primary_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();(source/'frozen-source.obj').write_text('fixture')
            sio.savemat(source/'frozen-edge-lengths.mat',{'edge_lengths':np.ones((1,3))})
            binary=root/'build/bin/GIF_controlled';binary.parent.mkdir(parents=True);binary.write_bytes(b'not executed')
            (binary.parent.parent/'bcp_torch_energy_native.oct').write_bytes(b'not executed')
            args=controlled.build_arg_parser().parse_args(['--frozen-input-dir',str(source),'--work-dir',str(root/'run'),
                '--native-bin',str(binary),'--octave',str(root/'octave'),'--device','cpu','--native-replay-stages'])
            def native(executable,arguments,env,folder,timeout,resume=False):
                folder.mkdir(parents=True,exist_ok=True);states=folder/'controller_states';states.mkdir(exist_ok=True)
                (states/'remesh_0001.mat').write_bytes(b'mocked validated snapshot')
                np.save(folder/'uv_final.npy',np.array([[0.,0.],[1.,0.],[0.,1.]]))
                return subprocess.CompletedProcess([],0,'GIF evaluator backend: torch (cpu, inprocess)','')
            manifest={'schema':'bcp-controlled-validation/1','diagnostic_only':True,'nonfinite_diagnostic_values':{}}
            trace={'exact':True,'passed':True,'first_arithmetic_difference':None,'first_envelope_failure':None,'first_branch_difference':None}
            fake_igl=types.SimpleNamespace(read_triangle_mesh=lambda p:(np.zeros((3,3)),np.array([[0,1,2]])))
            fake_compare=types.SimpleNamespace(compare_uv_outputs=lambda *a,**k:{'passed':True})
            fake_kernel=types.SimpleNamespace(isolate=mock.Mock(side_effect=RuntimeError('diagnostic failure')))
            with mock.patch.object(controlled,'checked_native',side_effect=native), \
                 mock.patch.object(controlled,'read_native_manifest',return_value=manifest), \
                 mock.patch.object(controlled,'compare_native_runs',return_value={'exact':True}), \
                 mock.patch.object(controlled,'validate_snapshot'), \
                 mock.patch.object(controlled,'compare_snapshots',return_value={'exact':True}), \
                 mock.patch.object(controlled,'replay_controller',return_value={'energy':2.}), \
                 mock.patch.object(controlled,'compare_controller_runs',return_value=trace), \
                 mock.patch.dict('sys.modules',{'igl':fake_igl,'compare':fake_compare,'controlled_kernel':fake_kernel}):
                report=controlled.run(args)
            self.assertTrue(report['primary_checks_passed'])
            self.assertTrue(report['kernel_isolation_failed'])
            self.assertFalse(report['ok'])
            self.assertNotIn('execution_error',report)
            self.assertEqual(report['runs']['remesh_0001']['kernel_isolation']['error'],'diagnostic failure')

    @staticmethod
    def write_step(folder, step=1, **overrides):
        state={'schema_version':1,'x_before':np.ones((2,1)),'fz':np.ones((2,1),complex),
               'fbz':np.zeros((2,1),complex),'E':2.,'G1':np.ones((2,1)),'H1':np.eye(2),
               'd1':np.ones((2,1)),'tMax':1.,'t':1.,'x_after':np.ones((2,1))*2,
               'primary_chol_p':0,'fallback':0}
        state.update(overrides)
        sio.savemat(folder/f'remesh_0001_step_{step:04d}.mat',{'state':state})

    def test_nonfinite_diagnostic_ranking_is_numerical_not_execution_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);np.save(folder/'ranking.npy',np.array([[np.inf]]))
            artifact={'name':'outlier_ranking_values','file':'ranking.npy','dtype':'<f8','shape':[1,1],
                      'stage':'outlier','index_space':'face','fortran_order':False}
            (folder/'manifest.json').write_text(json.dumps({'schema':'bcp-controlled-validation/1','diagnostic_only':True,
                'meta':{},'artifacts':[artifact]}))
            report=controlled.read_native_manifest(folder)
            self.assertEqual(report['nonfinite_diagnostic_values'],{'outlier_ranking_values':1})
            self.assertFalse(controlled.compare_native_runs(folder,folder)['exact'])
            artifact['name']='basis_values_real'
            (folder/'manifest.json').write_text(json.dumps({'schema':'bcp-controlled-validation/1','diagnostic_only':True,
                'meta':{},'artifacts':[artifact]}))
            with self.assertRaises(ValueError):controlled.read_native_manifest(folder)

    def test_missing_trace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';a.mkdir();b=Path(tmp)/'b';b.mkdir()
            self.assertFalse(controlled.compare_controller_runs(a,b)['passed'])

    def test_first_divergence_and_branch_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';a.mkdir();b=Path(tmp)/'b';b.mkdir()
            self.write_step(a);self.write_step(b,H1=2*np.eye(2),fallback=1)
            report=controlled.compare_controller_runs(a,b)
            self.assertFalse(report['passed'])
            self.assertEqual(report['first_arithmetic_difference']['field'],'H1')
            self.assertEqual(report['first_envelope_failure']['field'],'H1')
            self.assertEqual(report['first_branch_difference']['field'],'fallback')

    def test_exact_repeat_and_iteration_count_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';a.mkdir();b=Path(tmp)/'b';b.mkdir()
            self.write_step(a);self.write_step(b)
            self.assertTrue(controlled.compare_controller_runs(a,b)['exact'])
            self.write_step(b,step=2)
            report=controlled.compare_controller_runs(a,b)
            self.assertFalse(report['passed']);self.assertEqual(report['first_branch_difference']['reason'],'different_iteration_counts')

    def test_stop_reason_difference_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';a.mkdir();b=Path(tmp)/'b';b.mkdir()
            self.write_step(a);self.write_step(b)
            for folder,reason in ((a,'gradient_tolerance'),(b,'step_tolerance')):
                sio.savemat(folder/'remesh_0001_stop.mat',{'terminal':{'schema_version':1,'reason':reason}})
            result=controlled.compare_controller_runs(a,b)
            self.assertFalse(result['passed']);self.assertEqual(result['first_branch_difference']['reason'],'different_stop_reasons')


if __name__=='__main__':unittest.main()
