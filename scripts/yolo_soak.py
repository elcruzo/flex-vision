"""Continuous detector soak after measured thermal conditioning."""
import csv
import json
import subprocess
from inspection_soak import thermal_window
from yolo_load import FIELDS


def save(output, name, result, rows):
    filename=name+'.csv'
    with (output/filename).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)
    result['samples']=filename
    (output/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')


def run_soak(run,runner,consumer,sources,expected,references,detections,output):
    report={'kind':'continuous-detector-soak','status':'failed','conditioning':[],
            'policy':{'seconds':1800,'cameras':4,'fps_per_camera':60,'depth':2,
                      'thermal_range_c':2,'thermal_median_change_c':1,
                      'allocation_growth_allowance_bytes':1048576}}
    path=output/'telemetry.csv'
    with path.open('w') as telemetry:
        process=subprocess.Popen(['nvidia-smi',
            '--query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used',
            '--format=csv,noheader,nounits','-l','1'],stdout=telemetry)
        try:
            for attempt in range(2):
                block,rows=run('cpg',runner,consumer,sources,expected,references,detections,300,60,2,0)
                save(output,f'conditioning-{attempt}',block,rows);del rows
                thermal=thermal_window(path)
                report['conditioning'].append({'run':block,'thermal':thermal})
                if block['status']!='passed':raise AssertionError('Detector conditioning failed')
                if process.poll() is not None:raise RuntimeError('GPU telemetry stopped')
                print(json.dumps({'event':'detector_thermal_conditioning',**thermal}),flush=True)
                if thermal['stable']:break
            else:raise RuntimeError('Temperature did not stabilize within ten minutes')
            report['telemetry_start_row']=len(path.read_text().splitlines())
            block,rows=run('cpg',runner,consumer,sources,expected,references,detections,1800,60,2,0)
            save(output,'soak',block,rows);del rows
            report['run']=block
            early=[x for x in block['memory_samples'] if x['elapsed_s']<300]
            late=[x for x in block['memory_samples'] if 1500<=x['elapsed_s']<1800]
            if not early or not late:raise AssertionError('Missing soak memory windows')
            growth={key:max(x[key] for x in late)-max(x[key] for x in early)
                    for key in block['memory_before']}
            report['sampled_memory_ceiling_growth']=growth
            if process.poll() is not None:raise RuntimeError('GPU telemetry stopped')
            report['status']='passed' if block['status']=='passed' and all(v<=1048576 for v in growth.values()) else 'failed'
            return report
        except Exception as exc:
            report['error']=repr(exc);raise
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            (output/'soak-status.json').write_text(json.dumps(report,indent=2)+'\n')
