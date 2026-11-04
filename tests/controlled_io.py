"""Independent NumPy-format checks for the validation-only native I/O layer."""
from pathlib import Path
import json,os,subprocess,sys,tempfile
import numpy as np

binary=Path(sys.argv[1]).resolve();base=Path(sys.argv[2]).resolve();base.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='controlled-io-',dir=base) as tmp:
 root=Path(tmp);index=0
 for order in ('C','F'):
  for case in ('good','dtype','shape','nan','payload','duplicate'):
   index+=1;source=root/f'input-{index}';capture=root/f'capture-{index}';source.mkdir()
   data=np.array([[1.,2.,3.],[4.,5.,6.]],order=order)
   if case=='dtype':data=data.astype(np.float32)
   if case=='shape':data=data.reshape(3,2)
   if case=='nan':data[0,0]=np.nan
   np.save(source/'input.npy',data,allow_pickle=False)
   if case=='payload':
    with (source/'input.npy').open('ab') as file:file.write(b'garbage')
   env=dict(os.environ,BCP_CONTROLLED_DIR=str(capture),BCP_CONTROLLED_ORDER='index',
       BCP_CONTROLLED_REPLAY_STAGE='basis',BCP_CONTROLLED_REPLAY_DIR=str(source))
   run=subprocess.run([binary,'duplicate' if case=='duplicate' else 'good'],env=env,capture_output=True,text=True,timeout=20)
   if case=='good':
    assert run.returncode==0,(case,run.stderr)
    manifest=json.loads((capture/'manifest.json').read_text());assert manifest['diagnostic_only']
    assert np.array_equal(np.load(capture/'written.npy'),[[1,2,3],[4,5,6]])
    assert np.array_equal(np.load(capture/'ids.npy').reshape(-1),[0,7,9])
    again=subprocess.run([binary,'good'],env=env,capture_output=True,text=True,timeout=20)
    assert again.returncode!=0,'existing manifest overwritten'
   else:
    assert run.returncode!=0,(case,'malformed replay was accepted')
    if case=='duplicate':assert np.array_equal(np.load(capture/'written.npy'),[[1,2,3],[4,5,6]]),'duplicate replaced first payload'
print('Controlled native I/O: C/F order, dtype, shape, finite, payload, duplicate and fresh-directory checks passed')
