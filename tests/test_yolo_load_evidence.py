import csv
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_yolo_load import summarize


def test_raw_verifier_rejects_missing_arrival_and_failed_frame(tmp_path):
    runs=[]
    fields=['frame','lane','variant','arrival_s','status','submission_s','completion_s','arrival_to_completion_ms','gpu_pipeline_ms','correct']
    for fps in (60,400):
        for repeat in (0,1):
            for candidate in ('cpg','torch','cvcuda'):
                total=4*fps*30;name=f'{fps}-{repeat}-{candidate}.csv'
                with (tmp_path/name).open('w',newline='') as file:
                    writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
                    for index in range(total):
                        arrival=index/(4*fps)
                        row={'frame':index,'lane':(index+repeat)%4,'variant':(index//4)%2,'arrival_s':arrival,'status':'dropped_queue_full'}
                        if index<4 or index>=total-4:
                            row.update(status='completed',submission_s=arrival,completion_s=arrival+.001,arrival_to_completion_ms=1,gpu_pipeline_ms=.5,correct=True)
                        writer.writerow(row)
                runs.append({'fps_per_camera':fps,'repeat':repeat,'candidate':candidate,'status':'passed','failed_frames':[],
                             'seconds':30,'cameras':4,'depth':2,'samples':name,'offered':total,'completed':8,'dropped':total-8,
                             'max_pending':8,'elapsed_with_drain_s':31,'memory_before':{'torch_allocated':100},
                             'memory_after':{'torch_allocated':100},'memory_growth':{'torch_allocated':0},'memory_samples':[{'elapsed_s':1}]})
    (tmp_path/'results.json').write_text(json.dumps({'status':'passed','dirty':False,'runs':runs,'scope':'test','allocation_growth_allowance_bytes':1048576}))
    assert summarize(tmp_path)['status']=='verified'
    target=tmp_path/runs[0]['samples'];original=target.read_text()
    target.write_text('\n'.join(original.splitlines()[:-1])+'\n')
    with pytest.raises(ValueError,match='arrival records'):summarize(tmp_path)
    target.write_text(original.replace('True','False',1))
    with pytest.raises(ValueError,match='failed completed frame'):summarize(tmp_path)

    target.write_text(original)
    record=json.loads((tmp_path/'results.json').read_text())
    record['status']='failed'
    block=record['runs'][0];block['status']='failed'
    block['memory_after']['torch_allocated']=2457700
    block['memory_growth']['torch_allocated']=2457600
    (tmp_path/'results.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='passed load run'):summarize(tmp_path)
    audit=summarize(tmp_path,audit_failures=True)
    assert audit['status']=='verified_failed_experiment'
    assert len(audit['failed_blocks'])==1
