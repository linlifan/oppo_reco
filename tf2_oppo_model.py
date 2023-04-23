import os
import shutil
import argparse
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
#import pandas as pd
import json
import time
import sys
import csv
import types
from tensorflow.core.protobuf import rewriter_config_pb2

from tensorflow.python.client import timeline

import tf_graph_edit.graph_editor as ge

tf1.disable_eager_execution()

from datetime import datetime

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
#tf.logging.set_verbosity(tf.logging.DEBUG)


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "test_logs"

logdir = "{}/run-{}/".format(root_logdir, now)

export_savedmodel = 0 #1

profile =10
timeline_cnt = 0 #10
core = 8
BS = 256

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    help='model path',
                    dest='model_path',
                    default="./",
                    required=True)
parser.add_argument('--intra_threads', type=str,
                    help='number of threads to use',
                    required=False,
                    default="28",
                    dest='intra_threads')
parser.add_argument('--inter_threads', type=str,
                    help='number of threads to use',
                    required=False,
                    default="28",
                    dest='inter_threads')

parser.add_argument('--data-type', type=str,
                    help='model data type: bfloat16 or float32',
                    dest='data_type',
                    default="float32",
                    required=False)
parser.add_argument('--bs', type=int,
                    help='batch size',
                    dest='bs',
                    default=1000,
                    required=False)

args = parser.parse_args()

config = tf1.compat.v1.ConfigProto()

core = int(args.intra_threads)

inter_threads = int(args.inter_threads)

config = tf1.ConfigProto(intra_op_parallelism_threads=core, inter_op_parallelism_threads=inter_threads)

config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.OFF

if args.data_type != "float32":
   config.graph_options.rewrite_options.auto_mixed_precision_mkl = rewriter_config_pb2.RewriterConfig.ON


BS = args.bs

#inputs_dict = {}
#feed_dict = {}
#
#outputs_dict = {}
#fetch_ = []

def get_graph_input_output(args):
    
    with tf1.Session(config=config) as sess:
        g = tf1.Graph().as_default()
        
        inputs_dict = {}
        feed_dict = {}
       
        outputs_dict = {}
        fetch_ = []
    
        pb_file = args.model_path
        # pb file
        if pb_file.endswith('.pb'):
            with open(pb_file, "rb") as f:
                g_def = tf1.GraphDef()
                g_def.ParseFromString(f.read())
                _ = tf1.import_graph_def(g_def, name="")
            #sess.run('init_all_tables');
        
            for ops in sess.graph.get_operations() :
                if ops.type == "Placeholder":
                   in_tensor = graph.get_tensor_by_name(ops.name + ":0")
                   assert(in_tensor.dtype == 'float32' or in_tensor.dtype=='int64' or in_tensor.dtype=='int32')
                    
                   dim0 = in_tensor.shape.dims[0].value
                   assert ( dim0 == None) 
                   dim0 = BS
                   if in_tensor.shape.rank == 2:
                       shape = [dim0, in_tensor.shape.dims[1].value]
                   elif in_tensor.shape.rank == 1:
                       shape = [dim0]
                   elif in_tensor.shape.rank == 3:
                       shape = [dim0, in_tensor.shape.dims[1].value, in_tensor.shape.dims[2].value] 
                   #print(shape)
                   #continue
                   
                   inputs_dict[ops.name] = in_tensor
                   
                   if (in_tensor.dtype == "float32"):  #float point
                       feed_dict[in_tensor] = np.random.rand(*shape)
                   elif (in_tensor.dtype == "int64"):
                       feed_dict[in_tensor] = np.random.randint(1, 1000, size = shape)
                   elif (in_tensor.dtype == "int32"):
                       feed_dict[in_tensor] = np.random.randint(1, 1000, size = shape)
                   #elif (v.dtype == 7):
                   #    feed_dict[input_x] = np.array(['xxxxxx'])
                   else :
                       print(in_tensor.name + " :" + str(in_tensor.dtype))
                       pass 
     
       
            #add fetch tensor to fetch list and outputs_dict
            #fetch_.append(graph.get_tensor_by_name("score:0"))
            #outputs_dict["score"] = graph.get_tensor_by_name("score:0")
        
    
        else: 
        # Saved model
            #meta_graph = tf1.saved_model.loader.load(sess, ["serve"], pb_file) 
            #train graph for oppo 
            meta_graph = tf1.saved_model.loader.load(sess, ["train"], pb_file) 
            #sess.run('init_all_tables')
    
            #get inputs maps
            #inputs = meta_graph.signature_def['serving_default'].inputs
            #train graph for oppo 
            inputs = meta_graph.signature_def['serving'].inputs
            
            count = 0
               
            for k,v in inputs.items():
                count += 1
                
                dim0 = v.tensor_shape.dim[0].size
                if (v.tensor_shape.dim[0].size == -1) :
                    dim0 = BS
                shape = [dim0, v.tensor_shape.dim[1].size]
                
                print(shape)
                
                input_x = sess.graph.get_tensor_by_name(v.name)
                
                inputs_dict[v.name] = input_x 
               
                if (v.dtype == 1):  #float point
                    feed_dict[input_x] = np.random.rand(*shape)
                elif (v.dtype == 9):
                    feed_dict[input_x] = np.random.randint(1, 1000, size = shape)
                elif (v.dtype == 7):
                    feed_dict[input_x] = np.array(['xxxxxx'])
                else :
                    print(v.name + " :" + str(v.dtype))
                    pass 
            
            assert(count == len(feed_dict))
            #get outputs maps
            #outputs = meta_graph.signature_def['serving_default'].outputs
            
            #train graph for oppo 
            outputs = meta_graph.signature_def['serving'].outputs
            for k,v in outputs.items():
                fetch_.append(v.name)
                outputs_dict[v.name] = sess.graph.get_tensor_by_name(v.name) 
            
            print(fetch_)
            #exit()        
     
       
        graph = sess.graph
     
    return graph, inputs_dict, feed_dict, outputs_dict, fetch_ 



def run_inference(graph, feed_dict, fetch_): 
    with tf1.Session(graph=graph, config=config) as sess1:


       #sess1.run('init_all_tables')
       sess1.run(tf1.global_variables_initializer())
       # Warm up, two grapplers include the following test model 
       for _ in range(10):
           result = sess1.run(fetches = fetch_, feed_dict = feed_dict)
       #total += 1
       #sys.exit()

       # Test
       loops = 200
       print('start benchmark')
       start = time.time()
       for _ in range(loops):
           _ = sess1.run(fetches = fetch_, feed_dict= feed_dict)
       end = time.time()
       print ('Time of per loop: %f ms.' % ((end - start) / loops * 1000))

       if (profile > 0):
           options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,
                                                              python_tracer_level = 1,
                                                              device_tracer_level = 1)
           tf.profiler.experimental.start(logdir, options)
           for i in range(profile):
               sess1.run(fetches = fetch_, feed_dict= feed_dict)
           tf.profiler.experimental.stop()
           
           writer = tf1.summary.FileWriter(logdir, sess1.graph)
           writer.flush()
           writer.close()
       
       if timeline_cnt > 0:
           run_options  = tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE)
           run_metadata = tf1.RunMetadata()
 
           for i in range(timeline_cnt):
               sess1.run(fetch_, feed_dict, options=run_options, run_metadata=run_metadata)             
               tl = timeline.Timeline(run_metadata.step_stats)
               ctf = tl.generate_chrome_trace_format()
               with open('timeline.json', 'w') as f:
                   f.write(ctf)

def run_train(graph, inputs_dict, outputs_dict, feed_dict, fetch):
    
    #print(inputs_dict)
    
    diff_vars = []
    outputs = [] 
    
    for k,v in inputs_dict.items():
        diff_vars.append(v)
    
    diff_vars = diff_vars + fetch
 
    for k,v in outputs_dict.items():
        outputs.append(v)
         
    grad = tf1.gradients(outputs, diff_vars)
 
    grad_identity = [] 
    for v in grad :
        grad_identity.append(tf.identity(v))
    
    grad = grad_identity    
 
    with tf1.Session(graph=graph, config=config) as sess1:


       #sess1.run('init_all_tables')
       sess1.run(tf1.global_variables_initializer())
       # Warm up, two grapplers include the following test model 
       for _ in range(10):
           result = sess1.run(fetches = grad, feed_dict = feed_dict)
       
       # Test
       loops = 200
       print('start benchmark')
       start = time.time()
       for _ in range(loops):
           _ = sess1.run(fetches = grad, feed_dict= feed_dict)
       end = time.time()
       print ('Time of per loop: %f ms.' % ((end - start) / loops * 1000))

       if (profile > 0):
           options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,
                                                              python_tracer_level = 1,
                                                              device_tracer_level = 1)
           tf.profiler.experimental.start(logdir, options)
           for i in range(profile):
               sess1.run(fetches = grad, feed_dict= feed_dict)
           tf.profiler.experimental.stop()
           
           writer = tf1.summary.FileWriter(logdir, sess1.graph)
           writer.flush()
           writer.close()
       
       if timeline_cnt > 0:
           run_options  = tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE)
           run_metadata = tf1.RunMetadata()
 
           for i in range(timeline_cnt):
               sess1.run(grad, feed_dict, options=run_options, run_metadata=run_metadata)             
               tl = timeline.Timeline(run_metadata.step_stats)
               ctf = tl.generate_chrome_trace_format()
               with open('timeline.json', 'w') as f:
                   f.write(ctf)

def edit_graph(graph):


    param_tensors = []
    for ops in graph.get_operations() :
        if ops.type == "MatMul" or ops.type == "BiasAdd" :
           #print(ops)
           param_tensors.append(ops.inputs[1])
        if "batchnorm" in ops.name :
           for input in ops.inputs :
               if "ReadVariable" in input.name :
                   param_tensors.append(input) 
    
    #exit()
    #print(param_tensors)      
    print('{0:d} parameters'.format(len(param_tensors)))   
    
    const_var_name_pairs = []
    

    for c_tensor in param_tensors:
        #tensor = graph.get_tensor_by_name(t_name)
        name = c_tensor.name.split(':')[0]

        with tf1.Session() as sess:
            tensor_as_numpy_array = sess.run(c_tensor)
        
        var_shape = c_tensor.get_shape()
        var_name = '{}_turned_var'.format(name)
        var = tf1.get_variable(name=var_name, dtype='float32', shape=var_shape,  
                      initializer=tf.constant_initializer(tensor_as_numpy_array))
        
        const_var_name_pairs.append((name, var_name))
    
    for const_name, var_name in const_var_name_pairs:
        const_op = graph.get_operation_by_name(const_name)
        var_reader_op = graph.get_operation_by_name(var_name + '/Read/ReadVariableOp')
        ge.swap_outputs(ge.sgv(const_op), ge.sgv(var_reader_op))    
    
    #for ops in graph.get_operations() :
    #    if "deep_part_dense/target_0_fc3/MatMul/ReadVariableOp_turned_var" in ops.name:
    #       print(ops.name)

    new_vars = tf1.trainable_variables()  
     
     
    return new_vars

if __name__ == "__main__":

    graph, inputs_dict, feed_dict, outputs_dict, fetch = get_graph_input_output(args)

    #run_inference(graph, feed_dict, fetch)
   
    trainable_vars = edit_graph(graph) 
    
    run_train(graph, inputs_dict, outputs_dict, feed_dict, trainable_vars)

   
