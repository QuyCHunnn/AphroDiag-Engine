"""Experimental true row-batched acoustic graph; no production provider changes.

Global decoder and codec stay in the SDK. Each row has independent history,
uniform random draws and EOS. All sixteen local K/V steps remain inside graph.
"""
import time

from .frame_graph import install_frame_graph


def build_batched_frame(engine, batch_size):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    model = install_frame_graph(engine, build_only=True)
    graph = model.graph
    prefix = "jd_frame_"
    axis1 = "jd_batch_axis1"
    axes1 = "jd_batch_axes1"
    graph.initializer.extend([
        numpy_helper.from_array(np.array(1, np.int64), axis1),
        numpy_helper.from_array(np.array([1], np.int64), axes1),
    ])
    for tensor in graph.initializer:
        if tensor.name == prefix+"embedding_shape":
            tensor.CopyFrom(numpy_helper.from_array(np.array([batch_size, 1, engine.hidden], np.int64), tensor.name))
        elif tensor.name == prefix+"empty":
            tensor.CopyFrom(numpy_helper.from_array(
                np.zeros((batch_size, engine.nH_loc, 0, engine.hd_loc), np.float32), tensor.name))
    for value, shape in zip(graph.input, ([batch_size, 2, engine.hidden],
                                         [batch_size, *engine.audio_emb.shape[:2]],
                                         [batch_size, engine.n_vq])):
        value.CopyFrom(helper.make_tensor_value_info(value.name, value.type.tensor_type.elem_type, shape))
    graph.output[0].CopyFrom(helper.make_tensor_value_info(prefix+"codes", onnx.TensorProto.INT64, [batch_size, engine.n_vq]))
    graph.output[1].CopyFrom(helper.make_tensor_value_info(prefix+"eos", onnx.TensorProto.BOOL, [batch_size]))

    def attribute(node, name, value):
        for item in node.attribute:
            if item.name == name:
                item.i = value
                return
        node.attribute.append(helper.make_attribute(name, value))

    aliases = {prefix+f"c{i}_vector": prefix+f"c{i}_last_hidden" for i in range(engine.n_vq)}
    aliases[prefix+"slot0"] = prefix+"slot0_batch"
    nodes = []
    for node in graph.node:
        output = node.output[0]
        if output in aliases or output.startswith(prefix+"code_vector_"):
            continue
        node.input[:] = [aliases.get(name, name) for name in node.input]
        if output == prefix+"codes":
            node.input[:] = [prefix+f"c{i}_code" for i in range(engine.n_vq)]
            attribute(node, "axis", 1)
        elif output == prefix+"text_id":
            attribute(node, "axis", 1)
        elif output.startswith(prefix+"c") and "/" not in output:
            # Names ending here belong to the frame sampler, not copied SDK
            # attention nodes whose full names include '/c/...'.
            if output.endswith(("_history", "_uniform", "_cdf_end")):
                attribute(node, "axis", 1)
            if output.endswith(("_top_values", "_probabilities")):
                attribute(node, "axis", 1)
            if node.op_type == "CumSum":
                node.input[1] = axis1
            if output.endswith("_sum"):
                node.input[1] = axes1
                attribute(node, "keepdims", 1)
            if output.endswith("_sample"):
                attribute(node, "axis", 1)
                attribute(node, "keepdims", 1)
            if output.endswith("_code"):
                node.op_type = "GatherElements"
                attribute(node, "axis", 1)
            if output.endswith(("_uniform", "_cdf_end")):
                nodes.append(node)
                nodes.append(helper.make_node("Unsqueeze", [output, axes1], [output+"_column"]))
                continue
            if output.endswith("_selected"):
                node.input[1] += "_column"
            if output.endswith("_cdf_normalized"):
                node.input[1] += "_column"
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    graph.name = f"JellyDubAcousticFrameBatch{batch_size}"
    onnx.checker.check_model(model)
    return model


class BatchedFrameRuntime:
    def __init__(self, engine, batch_size, device="GPU"):
        if device not in {"CPU", "GPU"}:
            raise ValueError("Experimental batch runtime requires explicit CPU or GPU")
        self.batch_size = batch_size
        self.engine = engine
        model = build_batched_frame(engine, batch_size)
        self.calls = 0
        self.seconds = 0.0
        self.rows = 0
        self.histogram = {}
        if device == "CPU":
            import onnxruntime as ort
            self.session = ort.InferenceSession(model.SerializeToString(), engine._so, providers=["CPUExecutionProvider"])
            self._run = lambda feed: self.session.run(None, feed)
            self.devices = self.session.get_providers()
        else:
            import openvino as ov
            core = ov.Core()
            self.compiled = core.compile_model(core.read_model(model.SerializeToString()), "GPU", {
                "INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "LATENCY"})
            self.request = self.compiled.create_infer_request()
            self.devices = list(self.compiled.get_property("EXECUTION_DEVICES"))
            if not self.devices or any(not str(d).startswith("GPU") for d in self.devices):
                raise RuntimeError(f"Expected GPU execution: {self.devices}")
            def run(feed):
                self.request.infer(feed)
                return [self.request.get_output_tensor(i).data.copy() for i in (0, 1)]
            self._run = run

    def run(self, tokens, history, uniforms):
        import numpy as np
        count = len(tokens)
        if not 1 <= count <= self.batch_size:
            raise ValueError("Active rows must fit the compiled batch")
        arrays = [np.asarray(tokens, np.float32), np.asarray(history, bool), np.asarray(uniforms, np.float32)]
        if count < self.batch_size:
            arrays = [np.concatenate([a, np.repeat(a[-1:], self.batch_size-count, axis=0)], axis=0) for a in arrays]
        started = time.perf_counter()
        codes, eos = self._run(dict(zip(("jd_frame_tokens", "jd_frame_history", "jd_frame_uniforms"), arrays)))
        self.seconds += time.perf_counter()-started
        self.calls += 1
        self.rows += count
        self.histogram[count] = self.histogram.get(count, 0)+1
        return codes[:count].copy(), eos[:count].copy()
