import json
from pathlib import Path
import pytest
from scripts.run_staged_v33_suite import run_reporting


def test_report_preflight_renders_png_in_training_interpreter(tmp_path):
    run_reporting(tmp_path)
    state=json.loads((tmp_path/'pipeline_status.json').read_text())
    assert state['status']=='completed_preflight_reporting' and state['exit_code']==0
    result=json.loads((tmp_path/'report_preflight.log').read_text())
    assert result['status']=='passed' and result['backend'].lower()=='agg'


def test_missing_report_dependency_is_explicit_preflight_failure(tmp_path,monkeypatch):
    shadow=tmp_path/'shadow';shadow.mkdir()
    (shadow/'matplotlib.py').write_text("raise ModuleNotFoundError(\"No module named 'matplotlib'\")\n")
    monkeypatch.setenv('PYTHONPATH',str(shadow))
    with pytest.raises(SystemExit):run_reporting(tmp_path)
    state=json.loads((tmp_path/'pipeline_status.json').read_text())
    assert state['status']=='failed_preflight_reporting' and state['exit_code']!=0
    assert 'ModuleNotFoundError' in (tmp_path/'report_preflight.log').read_text()


def test_failed_report_keeps_completed_training_and_records_actual_stage(tmp_path):
    folder=tmp_path/'01_protocol_diagnostic';folder.mkdir()
    artifacts={'training_status.json':'{"status":"complete","step":1000}',
               'mechanisms.json':'{"step":100}', 'last.pt':'checkpoint sentinel'}
    for name,value in artifacts.items():(folder/name).write_text(value)
    (folder/'validation_step001000.json').write_text('incomplete JSON')
    with pytest.raises(SystemExit):run_reporting(tmp_path,folder.name,1)
    state=json.loads((tmp_path/'pipeline_status.json').read_text())
    assert state['status']=='failed_reporting' and state['experiment']==folder.name
    assert state['exit_code']!=0 and state['stage']==1
    assert 'JSONDecodeError' in (folder/'report.log').read_text()
    for name,value in artifacts.items():assert (folder/name).read_text()==value
