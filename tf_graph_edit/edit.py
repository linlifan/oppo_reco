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
#iop = tf.load_op_library('./libxsmm_matmul.so')

def factor(mb):
    ret = []
    for i in range(mb+1):
        comb = []
        if i == 0:
            continue
        if mb % i == 0:
            comb = [mb // i, i]
            ret.append(comb)
    return ret
   

def _is_worth_replace(a_shape, b_shape, transa, transb, dtype=tf.float32, config=None):
    """benchmark tf matmul op and custom matmul op
    """
    warm = 10
    loop = 100
     
    print('bench matmul with', a_shape, b_shape)
    batch_comb = factor(a_shape[0])
    
    print("batch size %d decomposing" % a_shape[0])
    print(batch_comb)
    #exit()
    
    # time_ list to store tf matmul, conv1x1 ...
    #time_ = [0] * (1 + len(batch_comb))
    
    time_  = 0;
    index_ = 0;
    inputs = []
    input_num = 1
    for i in range (input_num):
        if dtype == tf.float32:
            t = np.random.random(a_shape).astype("float32") - 0.5
            inputs.append(t)
        else:
            t = np.random.random(a_shape).astype("bfloat16") - 0.5
            inputs.append(t)
     

    for i in range(len(batch_comb)):
        w_n = batch_comb[i]
        #print("tuning combination")
        #print(w_n)

        g = tf.Graph()
        with g.as_default():
            a = tf.placeholder(shape=a_shape, dtype=dtype)
            b = tf.Variable(np.random.random(b_shape) - 0.5, dtype=dtype)
           
            c = tf.matmul(a, b, transpose_a = transa, transpose_b = transb)
            
            #conv_i = tf.reshape(a, [1,1,a_shape[0], a_shape[1]])
            conv_i = tf.reshape(a, [w_n[1], 1, w_n[0], a_shape[1]])
            conv_f = tf.reshape(b, [1, 1, b_shape[0], b_shape[1]])
            c2 = tf.nn.conv2d(conv_i, conv_f, padding="VALID") #iop.custom_matmul(a, b, transpose_a = transa, transpose_b = transb)
            
            sess = tf.Session(config=config)
            sess.run(tf.initialize_all_variables())
   
            for l in range(warm):
                sess.run(c, feed_dict={a: inputs[l % input_num]})
            start = time.time()
            for l in range(loop):
                sess.run(c, feed_dict={a: inputs[l % input_num]})
            t1 = time.time() - start

            for l in range(warm):
                sess.run(c2, feed_dict={a: inputs[l % input_num]})
            
            start = time.time()
            for l in range(loop):
                sess.run(c2, feed_dict={a: inputs[l % input_num]})
            t2 = time.time() - start
            
            print(t1, t2)

            if t2 < (t1 * 0.97):
                if index_ == 0 or t2 < time_:
                   time_ = t2
                   index_ = i
                   #print("index_ updating tp %d" % index_)
    
    conv_i_shape = []
    conv_f_shape = []
    
    if time_ > 0:
        conv_i_shape = [batch_comb[index_][1], 1, batch_comb[index_][0], a_shape[1]]
        conv_f_shape = [1, 1, b_shape[0], b_shape[1]]
    print("index %d" % index_) 
    print(conv_i_shape)
    print(conv_f_shape)
    if len(conv_i_shape) > 0 :
        return [conv_i_shape, conv_f_shape]
    else :
        return []

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
            
            #support plain format initially
            if transa == True or transb == True:
                continue

            a_shape = matmul_op.inputs[0].shape
            dtype = matmul_op.inputs[0].dtype
            if not a_shape.is_fully_defined():
                a_shape = sess.run(matmul_op.inputs[0], feed_dict=feed_dict).shape
            b_shape = matmul_op.inputs[1].shape
            
            if not b_shape.is_fully_defined():
                b_shape = sess.run(matmul_op.inputs[1], feed_dict=feed_dict).shape
            
            #skip gemv
            if b_shape[1] == 1:
                continue

            #with tf.Session(graph=graph, config=config) as sess1:
            if replace_all: 
                factor = 1 #125
                info[matmul_op.name]={'dtype':dtype, 
                                      'a_shape':a_shape,
                                      'b_shape':b_shape, 
                                      'a_shape_conv': [a_shape[0] // factor,1, factor, a_shape[1]], 'b_shape_conv': [1,1,b_shape[0], b_shape[1]]}
            else:
                shape_list = _is_worth_replace(a_shape, b_shape, transa, transb, dtype, config=config)
                if len(shape_list) > 0:
                    info[matmul_op.name]={'dtype':dtype, 'a_shape':a_shape, 'b_shape':b_shape, 'a_shape_conv': shape_list[0], 'b_shape_conv': shape_list[1]}

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

        a_shape_conv = matmul_info[matmul_name]['a_shape_conv']
        b_shape_conv = matmul_info[matmul_name]['b_shape_conv']
        dtype = matmul_info[matmul_name]['dtype']

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
                
                #fused_matmul = iop.custom_fused_matmul(tf.placeholder(shape=a_shape, dtype=tf.float32),
                #                                       tf.placeholder(shape=b_shape, dtype=tf.float32),
                #                                       tf.placeholder(shape=bias_shape, dtype=tf.float32),
                #                                       relu=True)
                #conv_i = tf.placeholder(shape=a_shape, dtype=tf.float32, name = "input")
                #conv_w = tf.placeholder(shape=b_shape, dtype=tf.float32, name = "weight")
                #conv_b = tf.placeholder(shape=bias_shape, dtype=tf.float32, name = "bias")
                
                conv_i_reshape = tf.reshape(tf.placeholder(shape=a_shape, dtype=dtype, name = "input"), a_shape_conv, name="input_reshaped")
                conv_w_reshape = tf.reshape(tf.placeholder(shape=b_shape, dtype=dtype, name = "weight"), b_shape_conv, name="weight_reshaped")
                conv_out = tf.nn.conv2d(conv_i_reshape, conv_w_reshape, padding="VALID", name="conv11")
                bias_out = tf.nn.bias_add(conv_out, tf.placeholder(shape=bias_shape, dtype=dtype, name = "bias"), name="add_bias")
                relu_out = tf.nn.relu(bias_out, name="act")
                sg_out = tf.reshape(relu_out, [a_shape[0], b_shape[1]], name="relu_reshaped")
                
                #new_subgraph = [fused_matmul.op]
                '''conv_i.op, conv_w.op, conv_b.op,''' 
                new_subgraph = [conv_i_reshape.op, conv_w_reshape.op, conv_out.op, bias_out.op, relu_out.op, sg_out.op]

                if relu.outputs[0] in fetches:
                    fetches.remove(relu.outputs[0])
                    #fetches.append(fused_matmul.op.outputs[0])
                    fetches.append(sg_out.op.outputs[0])
                    #print('replace fetch tensor ' + relu.outputs[0].name + ' with ' + fused_matmul.op.outputs[0].name)
                    print('replace fetch tensor ' + relu.outputs[0].name + ' with ' + sg_out.op.outputs[0].name)
                print('replace ' + matmul_op.name + ' + ' + biasadd.name + ' + ' + relu.name + ' with conv1x1_bias_relu')
            else:
                replace_type = 'fused_matmul'
                subgraph = [matmul_op, biasadd]
                #fused_matmul = iop.custom_fused_matmul(tf.placeholder(shape=a_shape, dtype=tf.float32),
                #                                       tf.placeholder(shape=b_shape, dtype=tf.float32),
                #                                       tf.placeholder(shape=bias_shape, dtype=tf.float32),
                #                                       relu=False)
                conv_i_reshape = tf.reshape(tf.placeholder(shape=a_shape, dtype=dtype, name = "input"), a_shape_conv, name="input_reshaped")
                conv_w_reshape = tf.reshape(tf.placeholder(shape=b_shape, dtype=dtype, name = "weight"), b_shape_conv, name="weight_reshaped")
                conv_out = tf.nn.conv2d(conv_i_reshape, conv_w_reshape, padding="VALID", name="conv11")
                bias_out = tf.nn.bias_add(conv_out, tf.placeholder(shape=bias_shape, dtype=dtype, name = "bias"), name="add_bias")
                sg_out = tf.reshape(bias_out, [a_shape[0], b_shape[1]], name="bias_out_reshaped")
                
                new_subgraph = [conv_i_reshape.op, conv_w_reshape.op, conv_out.op, bias_out.op, sg_out.op]

                if biasadd.outputs[0] in fetches:
                    fetches.remove(biasadd.outputs[0])
                    fetches.append(sg_out.op.outputs[0])
                    print('replace fetch tensor ' + biasadd.outputs[0].name + ' with ' + sg_out.op.outputs[0].name)
                print('replace ' + matmul_op.name + ' + ' + biasadd.name + ' with conv1x1_bias')
             
            # replace input tensors and output tensors
            sgv0 = ge.sgv(subgraph)
            sgv1 = ge.sgv(new_subgraph).remap_inputs([0,2,4])

        if replace_type == 'matmul':
            #matmul = iop.custom_matmul(tf.placeholder(shape=a_shape, dtype=tf.float32),
            #                           tf.placeholder(shape=b_shape, dtype=tf.float32),
            #                           transpose_a = transa, transpose_b = transb)
            
            subgraph = [matmul_op]
            
            conv_i_reshape = tf.reshape(tf.placeholder(shape=a_shape, dtype=dtype, name = "input"), a_shape_conv, name="input_reshaped")
            conv_w_reshape = tf.reshape(tf.placeholder(shape=b_shape, dtype=dtype, name = "weight"), b_shape_conv, name="weight_reshaped")
            conv_out = tf.nn.conv2d(conv_i_reshape, conv_w_reshape, padding="VALID", name="conv11")
            sg_out = tf.reshape(conv_out, [a_shape[0], b_shape[1]], name="conv_out_reshaped")
                
            new_subgraph = [conv_i_reshape.op, conv_w_reshape.op, conv_out.op, sg_out.op]

            if matmul_op.outputs[0] in fetches:
                fetches.remove(matmul_op.outputs[0])
                fetches.append(sg_out.op.outputs[0])
                print('replace fetch tensor ' + matmul_op.outputs[0].name + ' with ' + sg_out.op.outputs[0].name)
            print('replace ' + matmul_op.name + ' with conv1x1')

            # replace input tensors and output tensors
            sgv0 = ge.sgv(subgraph)
            sgv1 = ge.sgv(new_subgraph).remap_inputs([0,2])
            #print(sgv0)
            #print(sgv1)
        
        ge.swap_outputs(sgv0, sgv1)
        ge.swap_inputs(sgv0,sgv1)
        #ge.swap_ios(subgraph, new_subgraph)

        # replace control inputs
        controls = ge.util.ControlOutputs(graph)
        for op in subgraph:
            for control_output_op in controls.get(op):
                print("replace " + control_output_op.name + "'s control input")
                ge.remove_control_inputs(control_output_op, subgraph)
                ge.add_control_inputs(control_output_op, new_subgraph)


