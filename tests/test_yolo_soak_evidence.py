"""Reject false soak acceptance from incomplete or altered retained evidence."""
import csv
from datetime import datetime,timedelta
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_yolo_soak import summarize
from yolo_load import FIELDS


def test_complete_soak_evidence_and_faults(tmp_path):
    keys=['torch_allocated','torch_reserved','cupy_used','cupy_total']
    zeros=dict.fromkeys(keys,0)
    run={'candidate':'cpg','seconds':1800,'cameras':4,'fps_per_camera':60,'depth':2,
         'status':'passed','failed_frames':[],'max_pending':8,'elapsed_with_drain_s':1800,
         'offered':432000,'completed':24,'dropped':431976,'samples':'soak.csv',
         'memory_before':zeros,'memory_after':zeros,'memory_growth':zeros,
         'memory_samples':[{'elapsed_s':i/2,**zeros} for i in range(3600)]}
    report={'dirty':False,'status':'passed','soak':{'status':'passed','run':run,
            'telemetry_start_row':120,'sampled_memory_ceiling_growth':zeros}}
    def save():(tmp_path/'results.json').write_text(json.dumps(report))
    save()
    with (tmp_path/'soak.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader()
        for i in range(432000):
            arrival=i/240
            row={'frame':i,'lane':i%4,'variant':(i//4)%2,'arrival_s':arrival,'status':'dropped_queue_full'}
            if i%72000<4:
                row.update(status='completed',correct=True,submission_s=arrival,completion_s=arrival+.001,
                           gpu_pipeline_ms=.5,arrival_to_completion_ms=1.)
            writer.writerow(row)
    stamp=datetime(2026,10,9)
    telemetry=''.join((stamp+timedelta(seconds=i)).strftime('%Y/%m/%d %H:%M:%S.%f')+', 60, 1500, 100, 50, 2000\n' for i in range(1920))
    (tmp_path/'telemetry.csv').write_text(telemetry)
    assert summarize(tmp_path)['completed']==24
    run['offered']=431999;save()
    with pytest.raises(ValueError,match='Counts'):summarize(tmp_path)
    run['offered']=432000;run['memory_growth']={**zeros,'cupy_total':1};save()
    with pytest.raises(ValueError,match='allocator'):summarize(tmp_path)
    run['memory_growth']=zeros;save()
    (tmp_path/'telemetry.csv').write_text(telemetry[:-100])
    with pytest.raises((ValueError,IndexError)):summarize(tmp_path)
