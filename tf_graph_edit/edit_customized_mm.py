import time
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
#from tensorflow.contrib import graph_editor as ge
import tf_graph_edit.graph_editor as ge

"""functions to edit tf model graph

Usage example:
    # build model graph...

    # edit graph
    from tf_graph_edit import edit
    graph = tf.get_default_graph()
    matmul_info = edit.get_matmul_info(graph=graph, config=config, feed_dict=feed_dict)
    edit.edit_graph(matmul_info, graph=graph, fetches=fetches)

    # run model...
"""

# load custom matmul and fused_matmul ops from .so library file
iop = tf.load_op_library('./libxsmm_matmul.so')

def _is_worth_replace(a_shape, b_shape, transa, transb, sess):
    """benchmark tf matmul op and custom matmul op
    """
    warm = 10
    loop = 100
    print('bench matmul with', a_shape, b_shape)
    a = tf.placeholder(shape=a_shape, dtype=tf.float32)
    b = tf.Variable(np.random.random(b_shape) - 0.5, dtype=tf.float32)
    t = np.random.random(a_shape).astype(np.float32) - 0.5
    c = tf.matmul(a, b, transpose_a = transa, transpose_b = transb)
    c2 = iop.custom_matmul(a, b, transpose_a = transa, transpose_b = transb)

    sess.run(tf.initialize_all_variables())
    for _ in range(warm):
        sess.run(c, feed_dict={a: t})
    start = time.time()
    for l in range(loop):
        sess.run(c, feed_dict={a: t})
    t1 = time.time() - start

    for _ in range(warm):
        sess.run(c2, feed_dict={a: t})
    start = time.time()
    for l in range(loop):
        sess.run(c2, feed_dict={a: t})
    t2 = time.time() - start
    print(t1, t2)

    if t2 < t1:
        return True
    else:
        return False

def get_matmul_info(graph, config=None, feed_dict=None, replace_all=False):
    """get matmul ops and shapes info that will be replaced

    Args:
        graph: the graph to be edited
        config: session config to bench matmul performance
        feed_dict: may needed to inference matmul shapes
        replace_all: replace all matmul ops
    Returns:
        dict contains matmul op name and shapes info that will be replaced.
    """
    info = {}
    with tf.Session(graph=graph, config=config) as sess:
        matmul_ops = [op for op in sess.graph.get_operations() if op.type == 'MatMul']
        for matmul_op in matmul_ops:
            transa = matmul_op.get_attr("transpose_a")
            transb = matmul_op.get_attr("transpose_b")
            a_shape = matmul_op.inputs[0].shape
            if not a_shape.is_fully_defined():
                a_shape = sess.run(matmul_op.inputs[0], feed_dict=feed_dict).shape
            b_shape = matmul_op.inputs[1].shape
            if not b_shape.is_fully_defined():
                b_shape = sess.run(matmul_op.inputs[1], feed_dict=feed_dict).shape
            with tf.Session(graph=graph, config=config) as sess1:
                if replace_all or _is_worth_replace(a_shape, b_shape, transa, transb, sess1):
                    info[matmul_op.name]={'a_shape': a_shape, 'b_shape': b_shape}
    return info

def edit_graph(matmul_info, graph, fetches=[]):
    """edit graph, replace tf matmul with custom matmul, replace tf matmul+biasadd/matmul+biasadd+relu with custom fused_matmul

    Args:
        matmul_info: dict contains matmul op name and shapes info that will be replaced.
        graph: the graph to be edited
        fetches: list of fetch tensors, if a fetch tensor is an output of matmul op, it will be replaced
    """
    for matmul_name in matmul_info:
        matmul_op = graph.get_operation_by_name(matmul_name)
        out = matmul_op.outputs[0] # output tensor of MatMul
        transa = matmul_op.get_attr("transpose_a")
        transb = matmul_op.get_attr("transpose_b")
        a_shape = matmul_info[matmul_name]['a_shape']
        b_shape = matmul_info[matmul_name]['b_shape']

        replace_type = 'matmul'
        subgraph = []
        new_subgraph = []

        consuming_ops = ge.get_consuming_ops(out)
        # custom fused matmul does not support transposed inputs
        if transa == False and transb == False and len(consuming_ops) == 1 and consuming_ops[0].type == 'BiasAdd':
            biasadd = consuming_ops[0]
            bias_shape = biasadd.inputs[1].shape
            consuming_ops = ge.get_consuming_ops(biasadd.outputs[0])
            if len(consuming_ops) == 1 and consuming_ops[0].type == 'Relu':
                relu = consuming_ops[0]
                replace_type = 'fused_matmul_relu'
                subgraph = [matmul_op, biasadd, relu]
                fused_matmul = iop.custom_fused_matmul(tf.placeholder(shape=a_shape, dtype=tf.float32),
                                                       tf.placeholder(shape=b_shape, dtype=tf.float32),
                                                       tf.placeholder(shape=bias_shape, dtype=tf.float32),
                                                       relu=True)
                new_subgraph = [fused_matmul.op]
                if relu.outputs[0] in fetches:
                    fetches.remove(relu.outputs[0])
                    fetches.append(fused_matmul.op.outputs[0])
                    print('replace fetch tensor ' + relu.outputs[0].name + ' with ' + fused_matmul.op.outputs[0].name)
                print('replace ' + matmul_op.name + ' + ' + biasadd.name + ' + ' + relu.name + ' with fused_matmul_relu')
            else:
                replace_type = 'fused_matmul'
                subgraph = [matmul_op, biasadd]
                fused_matmul = iop.custom_fused_matmul(tf.placeholder(shape=a_shape, dtype=tf.float32),
                                                       tf.placeholder(shape=b_shape, dtype=tf.float32),
                                                       tf.placeholder(shape=bias_shape, dtype=tf.float32),
                                                       relu=False)
                new_subgraph = [fused_matmul.op]
                if biasadd.outputs[0] in fetches:
                    fetches.remove(biasadd.outputs[0])
                    fetches.append(fused_matmul.op.outputs[0])
                    print('replace fetch tensor ' + biasadd.outputs[0].name + ' with ' + fused_matmul.op.outputs[0].name)
                print('replace ' + matmul_op.name + ' + ' + biasadd.name + ' with fused_matmul')


        if replace_type == 'matmul':
            matmul = iop.custom_matmul(tf.placeholder(shape=a_shape, dtype=tf.float32),
                                       tf.placeholder(shape=b_shape, dtype=tf.float32),
                                       transpose_a = transa, transpose_b = transb)
            subgraph = [matmul_op]
            new_subgraph = [matmul.op]
            if matmul_op.outputs[0] in fetches:
                fetches.remove(matmul_op.outputs[0])
                fetches.append(matmul.op.outputs[0])
                print('replace fetch tensor ' + matmul_op.outputs[0].name + ' with ' + matmul.op.outputs[0].name)
            print('replace ' + matmul_op.name + ' with matmul')

        # replace input tensors and output tensors
        ge.swap_ios(subgraph, new_subgraph)

        # replace control inputs
        controls = ge.util.ControlOutputs(graph)
        for op in subgraph:
            for control_output_op in controls.get(op):
                print("replace " + control_output_op.name + "'s control input")
                ge.remove_control_inputs(control_output_op, subgraph)
                ge.add_control_inputs(control_output_op, new_subgraph)


