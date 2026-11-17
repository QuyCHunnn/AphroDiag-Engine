"""Experimental ONNX acoustic output-head fusion for the installed VieNeu SDK.

No vendor files are modified. Sampling and the SDK generation/babble guards stay
unchanged. This prototype must pass measured waveform and speed gates before use.
"""
from types import MethodType


def install_fused_heads(engine):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import onnxruntime as ort

    source = getattr(engine.sess_ac, "_model_path", None)
    if not source:
        raise ValueError("Acoustic session does not expose its source ONNX path")
    graph = onnx.load(source)
    hidden = graph.graph.output[0].name
    # Prefix every new graph name to avoid collisions with SDK-exported nodes.
    prefix = "jellydub_head_"
    if any(value.name.startswith(prefix) for value in graph.graph.initializer):
        raise ValueError("Graph already contains fused heads")
    weights = engine.audio_emb.astype(np.float32, copy=False)
    graph.graph.initializer.extend([
        numpy_helper.from_array(weights, prefix+"weights"),
        numpy_helper.from_array(np.array(-1, dtype=np.int64), prefix+"last"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), prefix+"zero"),
    ])
    graph.graph.input.append(helper.make_tensor_value_info(prefix+"channel", TensorProto.INT64, []))
    graph.graph.node.extend([
        helper.make_node("Gather", [hidden, prefix+"last"], [prefix+"last_hidden"], axis=1),
        helper.make_node("Gather", [prefix+"last_hidden", prefix+"zero"], [prefix+"vector"], axis=0),
        helper.make_node("Gather", [prefix+"weights", prefix+"channel"], [prefix+"weight"], axis=0),
        helper.make_node("Transpose", [prefix+"weight"], [prefix+"transposed"], perm=[1, 0]),
        helper.make_node("MatMul", [prefix+"vector", prefix+"transposed"], [prefix+"logits"]),
    ])
    graph.graph.output.append(helper.make_tensor_value_info(prefix+"logits", TensorProto.FLOAT, [weights.shape[1]]))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(graph.SerializeToString(), engine._so, providers=["CPUExecutionProvider"])
    original = engine._acoustic_frame

    def acoustic_frame(self, h, temperature, top_k, top_p, rep_pen, hist):
        tokens = np.stack([h[0].astype(np.float32), self.text_emb[self.sgs].astype(np.float32)])[None]
        feed = {"token_emb": tokens, "position_ids": np.array([[0, 1]], np.int64)}
        feed.update(self._empty_past())

        def step(channel, inputs):
            inputs[prefix+"channel"] = np.array(channel, dtype=np.int64)
            output = session.run(None, inputs)
            # Existing SDK splits only the known K/V range, ignoring our last output.
            keys, values = self._split_past(output, self.L_loc)
            previous = hist[channel] if hist is not None else None
            code = self._sample(output[-1], temperature, top_k, top_p, rep_pen, previous)
            if hist is not None:
                hist[channel].add(code)
            return output[0], keys, values, code

        hidden, pk, pv, code = step(0, feed)
        slot0 = hidden[0, 0]
        codes = [code]
        for channel in range(1, self.n_vq):
            embedding = self.audio_emb[channel-1][codes[-1]].astype(np.float32)
            feed = {"token_emb": embedding.reshape(1, 1, self.hidden),
                    "position_ids": np.array([[channel+1]], np.int64)}
            feed.update(self._past_feed(pk, pv))
            _, pk, pv, code = step(channel, feed)
            codes.append(code)
        eos = int((slot0.astype(np.float32) @ self.text_emb.T).argmax()) == self.eos_speech
        return codes, eos

    engine._acoustic_frame = MethodType(acoustic_frame, engine)
    return {"original": original, "session": session,
            "providers": session.get_providers(), "scope": "acoustic audio heads; CPU sampling unchanged"}
