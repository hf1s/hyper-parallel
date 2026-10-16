import json,pathlib,shutil,sys,numpy as np
meta=json.load(sys.stdin)
root=pathlib.Path(sys.argv[1])
assert str(root).startswith('/workspace/zth-jt-precision-scripts/reference/inputs/')
root.mkdir(parents=True,exist_ok=False)
(root/'tensors').mkdir()
order=meta['param_order.json']['grad_dump_order']
shapes={item['name']:item['shape'] for item in meta['param_shapes.json']['optimizer']}
assert len(order)==46 and len(set(order))==46
records=[]
for index,name in enumerate(order):
    source=pathlib.Path('/dev/shm/mfcap/mfgd/grad')/f'g_float32_{34+index}.npy'
    arr=np.load(source,mmap_mode='r',allow_pickle=False)
    assert arr.dtype==np.float32 and list(arr.shape)==shapes[name],(name,source,arr.dtype,arr.shape,shapes[name])
    relative=f'tensors/grad-{index:02d}.npy'
    shape=list(arr.shape)
    arr._mmap.close()
    shutil.copyfile(source,root/relative)
    records.append(dict(step=301,stage='post_backward',kind='grad',name=name,dtype='float32',shape=shape,path=relative,storage_dtype='float32',encoding='native',backend='mf',rank=0,occurrence=0,bitwise_original=True,source=str(source),source_conversion='none (source fp32)',grad_dump_index=index))
    if (index+1)%10==0 or index==45:
        print('MF_REFERENCE_IMPORTED',index+1,'/46',flush=True)
for stage,kind,name,source in (
    ('loss','loss','loss','/dev/shm/mfcap/mfgd/scalars/train_loss_float32_277.npy'),
    ('post_clip','scalar','total_norm','/dev/shm/mfcap/mfgd/scalars/global_norm_float32_80.npy')):
    arr=np.load(source,mmap_mode='r',allow_pickle=False)
    assert arr.dtype==np.float32 and arr.size==1
    relative='tensors/'+name+'.npy'
    print('MF_REFERENCE_SCALAR',name,float(arr.reshape(-1)[0]),flush=True)
    shape=list(arr.shape)
    arr._mmap.close()
    shutil.copyfile(source,root/relative)
    records.append(dict(step=301,stage=stage,kind=kind,name=name,dtype='float32',shape=shape,path=relative,storage_dtype='float32',encoding='native',backend='mf',rank=0,occurrence=0,bitwise_original=True,source=source))
for name,data in meta.items():
    with (root/name).open('x') as f:json.dump(data,f,indent=2)
with (root/'records.jsonl').open('x') as f:
    for record in records:f.write(json.dumps(record)+'\n')
with (root/'provenance.json').open('x') as f:
    json.dump(dict(gradient_source='/dev/shm/mfcap/mfgd/grad',gradient_ids=[34,79],mapping_source='/workspace/align-work/MFInstrumentedStep/artifacts/param_order.json',mapping_field='grad_dump_order',stage='grad-reducer output before clipping, loss_scale=1',reference_inventory='/home/zth/mf-state/INVENTORY.md',training_run_started=False,hashes_or_checksums_calculated=False),f,indent=2)
print('MF_REFERENCE_READY',root,'records',len(records),flush=True)
