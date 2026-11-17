import json
import pytest
from jellydub.performance_trace import span


def test_trace_preserves_failure_and_pairs_events(tmp_path,monkeypatch):
    output=tmp_path/'trace.jsonl';monkeypatch.setenv('JELLYDUB_TRACE_FILE',str(output))
    with pytest.raises(ValueError,match='original'):
        with span('generation',batch=4):raise ValueError('original')
    events=[json.loads(s) for s in output.read_text().splitlines()]
    assert [e['event'] for e in events]==['begin','end']
    assert events[0]['span']==events[1]['span'] and events[1]['status']=='failed'
    assert events[1]['seconds']>=0 and events[0]['metadata']=={'batch':4}


def test_trace_io_failure_does_not_fail_pipeline(tmp_path,monkeypatch):
    monkeypatch.setenv('JELLYDUB_TRACE_FILE',str(tmp_path/'missing/trace.jsonl'))
    with span('inference'):pass
