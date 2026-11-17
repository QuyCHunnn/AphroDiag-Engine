"""Experimental single-call acoustic frame graph (16 autoregressive codebooks).

Sampling stays stochastic with externally supplied uniform draws and history.
Default SDK parameters only; other configurations use the original SDK method.
"""
from copy import deepcopy
from types import MethodType


def install_frame_graph(engine, *, device="CPU", precision="f32", build_only=False):
    import numpy as np
    import onnx
    from onnx import TensorProto as T, helper as h, numpy_helper as nh
    source = onnx.load(engine.sess_ac._model_path)
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for node in source.graph.node for a in node.attribute):
        raise ValueError("Nested acoustic subgraphs require an explicit converter")
    nodes, initializers = [], list(source.graph.initializer)
    shared = {value.name for value in initializers}
    prefix = "jd_frame_"
    def constant(name, value):
        name = prefix+name
        initializers.append(nh.from_array(np.asarray(value), name))
        return name
    def op(kind, inputs, name, **attrs):
        output = prefix+name
        nodes.append(h.make_node(kind, inputs, [output], **attrs))
        return output
    zero = constant("zero", np.array(0, np.int64))
    last = constant("last", np.array(-1, np.int64))
    axis0 = constant("axis0", np.array([0], np.int64))
    k = constant("k", np.array([25], np.int64))
    temp = constant("temperature", np.array(.8, np.float32))
    penalty = constant("penalty", np.array(1.2, np.float32))
    nucleus = constant("nucleus", np.array(.95, np.float32))
    zero_float = constant("zero_float", np.array(0, np.float32))
    shape = constant("embedding_shape", np.array([1, 1, engine.hidden], np.int64))
    empty = constant("empty", np.zeros((1, engine.nH_loc, 0, engine.hd_loc), np.float32))
    channels = []
    first_hidden = None
    previous_outputs = None
    for channel in range(engine.n_vq):
        label = f"c{channel}_"
        channel_index = constant(label+"index", np.array(channel, np.int64))
        weights = constant(label+"weights", engine.audio_emb[channel].T.copy())
        position = constant(label+"position", np.array([[0, 1]] if channel == 0 else [[channel+1]], np.int64))
        if channel == 0:
            embedding = prefix+"tokens"
        else:
            table = constant(label+"embedding_table", engine.audio_emb[channel-1])
            vector = op("Gather", [table, channels[-1]], label+"embedding", axis=0)
            embedding = op("Reshape", [vector, shape], label+"tokens")
        mapping = {"token_emb": embedding, "position_ids": position}
        for layer in range(engine.L_loc):
            mapping[f"past_k_{layer}"] = empty if previous_outputs is None else previous_outputs[1+layer]
            mapping[f"past_v_{layer}"] = empty if previous_outputs is None else previous_outputs[1+engine.L_loc+layer]
        def rename(name):
            return mapping.get(name, name if name in shared or not name else prefix+label+name)
        for node in source.graph.node:
            copied = deepcopy(node)
            copied.input[:] = [rename(name) for name in node.input]
            copied.output[:] = [rename(name) for name in node.output]
            if copied.name:
                copied.name = prefix+label+copied.name
            nodes.append(copied)
        previous_outputs = [rename(value.name) for value in source.graph.output]
        hidden = previous_outputs[0]
        if first_hidden is None:
            first_hidden = hidden
        vector = op("Gather", [hidden, last], label+"last_hidden", axis=1)
        vector = op("Gather", [vector, zero], label+"vector", axis=0)
        logits = op("MatMul", [vector, weights], label+"logits")
        seen = op("Gather", [prefix+"history", channel_index], label+"history", axis=0)
        negative = op("Less", [logits, zero_float], label+"negative")
        multiplied = op("Mul", [logits, penalty], label+"negative_penalty")
        divided = op("Div", [logits, penalty], label+"positive_penalty")
        adjusted = op("Where", [negative, multiplied, divided], label+"penalty")
        logits = op("Where", [seen, adjusted, logits], label+"adjusted")
        logits = op("Div", [logits, temp], label+"scaled")
        values, indices = prefix+label+"top_values", prefix+label+"top_indices"
        nodes.append(h.make_node("TopK", [logits, k], [values, indices], axis=0, largest=1, sorted=1))
        probabilities = op("Softmax", [values], label+"probabilities", axis=0)
        before = op("CumSum", [probabilities, zero], label+"before", exclusive=1)
        keep = op("Less", [before, nucleus], label+"keep")
        kept = op("Where", [keep, probabilities, zero_float], label+"kept")
        total = op("ReduceSum", [kept, axis0], label+"sum", keepdims=0)
        probabilities = op("Div", [kept, total], label+"normalized")
        cdf = op("CumSum", [probabilities, zero], label+"cdf")
        end = op("Gather", [cdf, last], label+"cdf_end", axis=0)
        cdf = op("Div", [cdf, end], label+"cdf_normalized")
        uniform = op("Gather", [prefix+"uniforms", channel_index], label+"uniform", axis=0)
        greater = op("Greater", [cdf, uniform], label+"selected")
        selected = op("Cast", [greater], label+"selected_int", to=T.INT64)
        selected = op("ArgMax", [selected], label+"sample", axis=0, keepdims=0)
        channels.append(op("Gather", [indices, selected], label+"code", axis=0))
    code_vectors = [op("Unsqueeze", [code, axis0], f"code_vector_{i}") for i, code in enumerate(channels)]
    codes = op("Concat", code_vectors, "codes", axis=0)
    slot0 = op("Gather", [first_hidden, zero], "slot0_batch", axis=1)
    slot0 = op("Gather", [slot0, zero], "slot0", axis=0)
    text_weight = constant("text_weight", engine.text_emb.T.copy())
    text_logits = op("MatMul", [slot0, text_weight], "text_logits")
    text_id = op("ArgMax", [text_logits], "text_id", axis=0, keepdims=0)
    eos_id = constant("eos_id", np.array(engine.eos_speech, np.int64))
    eos = op("Equal", [text_id, eos_id], "eos")
    graph = h.make_graph(nodes, "JellyDubAcousticFrame", [
        h.make_tensor_value_info(prefix+"tokens", T.FLOAT, [1, 2, engine.hidden]),
        h.make_tensor_value_info(prefix+"history", T.BOOL, list(engine.audio_emb.shape[:2])),
        h.make_tensor_value_info(prefix+"uniforms", T.FLOAT, [engine.n_vq]),
    ], [h.make_tensor_value_info(codes, T.INT64, [engine.n_vq]),
        h.make_tensor_value_info(eos, T.BOOL, [])], initializer=initializers)
    model = h.make_model(graph, opset_imports=list(source.opset_import))
    model.ir_version = source.ir_version
    onnx.checker.check_model(model)
    if build_only:
        return model
    if device == "CPU":
        import onnxruntime as ort
        session = ort.InferenceSession(model.SerializeToString(), engine._so, providers=["CPUExecutionProvider"])
        run = lambda feed: session.run(None, feed)
        devices = session.get_providers()
    else:
        import openvino as ov
        core = ov.Core()
        compiled = core.compile_model(core.read_model(model.SerializeToString()), device,
                                     {"INFERENCE_PRECISION_HINT": precision, "PERFORMANCE_HINT": "LATENCY"})
        request = compiled.create_infer_request()
        def run(feed):
            request.infer(feed)
            return [request.get_output_tensor(i).data.copy() for i in (0, 1)]
        devices = list(compiled.get_property("EXECUTION_DEVICES"))
    original = engine._acoustic_frame
    def frame(self, hidden, temperature, top_k, top_p, rep_pen, hist):
        if (temperature, top_k, top_p, rep_pen) != (.8, 25, .95, 1.2):
            return original(hidden, temperature, top_k, top_p, rep_pen, hist)
        history = np.zeros(self.audio_emb.shape[:2], dtype=bool)
        if hist is not None:
            for channel in range(self.n_vq):
                previous = list(hist[channel])
                history[channel, previous] = True
        tokens = np.stack([hidden[0].astype(np.float32), self.text_emb[self.sgs].astype(np.float32)])[None]
        codes, ended = run({prefix+"tokens": tokens, prefix+"history": history,
                            prefix+"uniforms": np.random.random(self.n_vq).astype(np.float32)})
        values = [int(c) for c in codes]
        if hist is not None:
            for channel, value in enumerate(values):
                hist[channel].add(value)
        return values, bool(ended)
    engine._acoustic_frame = MethodType(frame, engine)
    return {"original": original, "devices": devices,
            "acoustic_calls_per_frame": 1, "nodes": len(nodes)}
