"""Experiment: make the export digestible by X2Paddle (opset<=15, no LayerNormalization)."""
import sys, onnx, numpy as np
from onnx import helper, numpy_helper, TensorProto
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
# 1. fold Constant nodes into initializers
consts = {}
new_nodes = []
for n in g.node:
    if n.op_type == 'Constant' and n.attribute and n.attribute[0].name == 'value':
        t = onnx.helper.get_attribute_value(n.attribute[0]); t.name = n.output[0]
        g.initializer.append(t); consts[n.output[0]] = t
    else:
        new_nodes.append(n)
del g.node[:]; g.node.extend(new_nodes)
# 2. rewrite LayerNormalization -> primitives
new_nodes = []
k = 0
for n in g.node:
    if n.op_type != 'LayerNormalization':
        new_nodes.append(n); continue
    x, scale, bias = list(n.input) + [None] * (3 - len(n.input))
    eps = next((a.f for a in n.attribute if a.name == 'epsilon'), 1e-5)
    axis = next((a.i for a in n.attribute if a.name == 'axis'), -1)
    y = n.output[0]; p = f'ln{k}_'; k += 1
    eps_t = numpy_helper.from_array(np.array(eps, dtype=np.float32), p + 'eps'); g.initializer.append(eps_t)
    new_nodes += [
        helper.make_node('ReduceMean', [x], [p + 'mean'], axes=[axis], keepdims=1),
        helper.make_node('Sub', [x, p + 'mean'], [p + 'xc']),
        helper.make_node('Mul', [p + 'xc', p + 'xc'], [p + 'sq']),
        helper.make_node('ReduceMean', [p + 'sq'], [p + 'var'], axes=[axis], keepdims=1),
        helper.make_node('Add', [p + 'var', p + 'eps'], [p + 'vare']),
        helper.make_node('Sqrt', [p + 'vare'], [p + 'std']),
        helper.make_node('Div', [p + 'xc', p + 'std'], [p + 'norm']),
        helper.make_node('Mul', [p + 'norm', scale], [p + 'scaled' if bias else y]),
    ]
    if bias:
        new_nodes.append(helper.make_node('Add', [p + 'scaled', bias], [y]))
del g.node[:]; g.node.extend(new_nodes)
# 3. ReduceMean/ReduceSum/ReduceMax with axes as input -> attribute (opset<18 form)
init = {t.name: t for t in g.initializer}
new_nodes = []
for n in g.node:
    if n.op_type in ('ReduceMean', 'ReduceMax') and len(n.input) == 2 and n.input[1] in init:
        axes = numpy_helper.to_array(init[n.input[1]]).tolist()
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        attrs.pop('noop_with_empty_axes', None)
        n2 = helper.make_node(n.op_type, [n.input[0]], list(n.output), axes=axes, **attrs)
        new_nodes.append(n2)
    else:
        new_nodes.append(n)
del g.node[:]; g.node.extend(new_nodes)
ops = sorted({n.op_type for n in g.node}); print('ops', ops)
del m.opset_import[:]; m.opset_import.append(helper.make_opsetid('', 15))
# Split with num_outputs / Pad with axes etc. would break; check
onnx.checker.check_model(m)
onnx.save(m, dst)
import onnxruntime as ort
s = ort.InferenceSession(dst, providers=['CPUExecutionProvider'])
s0 = ort.InferenceSession(src, providers=['CPUExecutionProvider'])
x = (np.random.rand(2, 3, 96, 500).astype(np.float32) * 2 - 1)
x[1, :, :, 300:] = 0
a = s.run(None, {'x': x})[0]; b = s0.run(None, {'x': x})[0]
print('rewritten vs original max diff', float(np.abs(a - b).max()))
