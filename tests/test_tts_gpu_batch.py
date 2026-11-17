"""Row mapping, failure propagation and request-local sampling for GPU prototype."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from jellydub.tts.gpu_batch_runtime import AcousticBatcher,clone_model_for_request


def test_coalesces_independent_requests_and_maps_outputs():
    calls=[]
    class Runtime:
        batch_size=4
        def run(self,tokens,histories,uniforms):
            calls.append(list(tokens))
            return np.array([[x*10] for x in tokens]),np.array([x%2 for x in tokens],bool)
    batcher=AcousticBatcher(Runtime(),max_wait_seconds=.1)
    barrier=threading.Barrier(4)
    def call(i):
        barrier.wait()
        codes,eos=batcher.submit(i,None,None)
        return int(codes[0]),eos
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(call,range(4)))
    finally:batcher.close()
    assert results==[(0,False),(10,True),(20,False),(30,True)]
    assert len(calls)==1 and sorted(calls[0])==[0,1,2,3]


def test_batch_failure_is_reported_without_hanging_other_rows():
    class Runtime:
        batch_size=2
        def run(self,*args):raise ValueError('runtime failed')
    batcher=AcousticBatcher(Runtime())
    try:
        with pytest.raises(ValueError,match='runtime failed'):batcher.submit(None,None,None)
        with pytest.raises(RuntimeError,match='worker failed'):batcher.submit(None,None,None)
    finally:batcher.close()


def test_request_rng_does_not_leak_between_cues_or_global_numpy():
    model=SimpleNamespace(engine=SimpleNamespace())
    a=clone_model_for_request(model,11);b=clone_model_for_request(model,12);reference=clone_model_for_request(model,11)
    logits=np.linspace(-1,2,32,dtype=np.float32)
    state=np.random.get_state()
    expected=[reference.engine._sample(logits,.8,25,.95,1.2,None) for _ in range(20)]
    actual=[]
    for _ in range(20):
        b.engine._sample(logits,.8,25,.95,1.2,None)
        actual.append(a.engine._sample(logits,.8,25,.95,1.2,None))
    after=np.random.get_state()
    assert actual==expected
    assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]


def test_batched_graph_preserves_row_isolation_and_partial_batch(tmp_path):
    onnx=pytest.importorskip('onnx');pytest.importorskip('onnxruntime')
    from onnx import helper as h,TensorProto as T
    import onnxruntime as ort
    from jellydub.tts.batched_frame_graph import BatchedFrameRuntime
    source=h.make_model(h.make_graph([
        h.make_node('Identity',['token_emb'],['hidden']),
        h.make_node('Identity',['past_k_0'],['present_k_0']),
        h.make_node('Identity',['past_v_0'],['present_v_0'])], 'toy', [
        h.make_tensor_value_info('token_emb',T.FLOAT,[1,'S',4]),
        h.make_tensor_value_info('position_ids',T.INT64,[1,'S']),
        h.make_tensor_value_info('past_k_0',T.FLOAT,[1,1,'P',4]),
        h.make_tensor_value_info('past_v_0',T.FLOAT,[1,1,'P',4])], [
        h.make_tensor_value_info('hidden',T.FLOAT,[1,'S',4]),
        h.make_tensor_value_info('present_k_0',T.FLOAT,[1,1,'P',4]),
        h.make_tensor_value_info('present_v_0',T.FLOAT,[1,1,'P',4])]),opset_imports=[h.make_opsetid('',17)])
    source.ir_version=9;path=tmp_path/'toy.onnx';onnx.save(source,path)
    rng=np.random.RandomState(44)
    engine=SimpleNamespace(sess_ac=SimpleNamespace(_model_path=str(path)),hidden=4,nH_loc=1,hd_loc=4,L_loc=1,
                           n_vq=2,audio_emb=rng.normal(size=(2,32,4)).astype(np.float32),text_emb=rng.normal(size=(5,4)).astype(np.float32),eos_speech=2,_so=ort.SessionOptions())
    engine._so.intra_op_num_threads=1
    runtime=BatchedFrameRuntime(engine,2,'CPU')
    tokens=rng.normal(size=(2,2,4)).astype(np.float32);history=np.zeros((2,2,32),bool);uniforms=np.array([[.23,.56],[.45,.71]],np.float32)
    codes,eos=runtime.run(tokens,history,uniforms)
    tokens[1]*=-4;history[1,:,:10]=True;uniforms[1]=[.1,.9]
    changed,ended=runtime.run(tokens,history,uniforms)
    assert np.array_equal(codes[0],changed[0]) and eos[0]==ended[0]
    partial,partial_end=runtime.run(tokens[:1],history[:1],uniforms[:1])
    assert np.array_equal(codes[:1],partial) and np.array_equal(eos[:1],partial_end)


@pytest.mark.parametrize('seconds', [-.1, float('nan'), float('inf')])
def test_invalid_batch_wait_is_rejected(seconds):
    with pytest.raises(ValueError,match='finite'):
        AcousticBatcher(SimpleNamespace(batch_size=2),max_wait_seconds=seconds)


def test_closed_batch_worker_rejects_new_requests():
    batcher=AcousticBatcher(SimpleNamespace(batch_size=2))
    batcher.close()
    with pytest.raises(RuntimeError,match='closed'):
        batcher.submit(None,None,None)
