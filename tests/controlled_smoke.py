"""Exercise the full controlled runner, accepting a reported numerical mismatch, never execution failure."""
import json,os
from pathlib import Path
import shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));from compare import generate_curved_disk_mesh
binary=Path(sys.argv[1]).resolve();base=Path(sys.argv[2]).resolve();base.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='controlled-smoke-',dir=base) as tmp:
 folder=Path(tmp);mesh=folder/'mesh.obj';generate_curved_disk_mesh(mesh,grid_size=8)
 device=os.environ.get('BCP_TORCH_REFERENCE_DEVICE','cpu')
 command=[sys.executable,str(ROOT/'scripts/controlled_compare.py'),'--input',str(mesh),'--target-faces','40',
          '--work-dir',str(folder/'run'),'--native-bin',str(binary),'--device',device,'--max-remeshes','1','--timeout','120']
 result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=360)
 assert result.returncode in (0,1),(result.stdout,result.stderr)
 report=json.loads((folder/'run/report.json').read_text());assert 'execution_error' not in report,report
 assert report['measurement_scope']=='controlled_validation_with_diagnostic_io'
 assert report['native_repeat_exact'] and report['native_fresh_repeat_exact'] and report['native_replays_exact'],report
 assert report['requested_diagnostics_succeeded'] and not report.get('kernel_isolation_failed'),report
 assert not report['production_defaults_modified'] and not report['historical_golden_gates_modified']
 resumed=subprocess.run(command+['--resume'],cwd=ROOT,capture_output=True,text=True,timeout=120)
 assert resumed.returncode==result.returncode,(resumed.stdout,resumed.stderr)
 assert json.loads((folder/'run/report.json').read_text())['native_replays_exact']
 print('Controlled runner capture/replay/resume completed; strict numerical outcome:',report['ok'])
