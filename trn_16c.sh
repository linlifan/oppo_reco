
#TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_REMOVE="" 
#TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_INFERLIST_REMOVE="Sum" 
#TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_ADD="Sum"   




#DNNL_VERBOSE=0 numactl -C 0-7 python tf2_oppo_model.py --model savedmodel/ --bs 64 --intra_threads 8 --inter_threads=8 --data-type bfloat16 --training 0
#DNNL_VERBOSE=0 numactl -C 8-15 python tf2_oppo_model.py --model savedmodel/ --bs 64 --intra_threads 8 --inter_threads=8 --data-type bfloat16 --training 0
#DNNL_VERBOSE=0 numactl -C 16-23 python tf2_oppo_model.py --model savedmodel/ --bs 64 --intra_threads 8 --inter_threads=8 --data-type bfloat16 --training 0
#DNNL_VERBOSE=0 numactl -C 24-31 python tf2_oppo_model.py --model savedmodel/ --bs 64 --intra_threads 8 --inter_threads=8 --data-type bfloat16 --training 0

start_core=0



run_16c() {
# For model1

echo "$1 test 16 core 3 instances start with $2 , training $3 \n"

# 16 cores
#end_core=$(($start_core+15))
for ((i = 0; i<3; i++));
do
    offset=$(expr $i \* 16)
    start_=$(($start_core + $offset))
    end_=$(($start_+15))
    echo "$i start $start_  end $end_"
    #numactl -C $start_-$end_ python $1 --model savedmodel/ --bs $4 --intra_threads 16 --inter_threads 16 --data-type $2 --training $3 &
    ONEDNN_DEFAULT_FPMATH_MODE=BF16 numactl -C $start_-$end_ python $1 --model savedmodel/ --bs $4 --intra_threads 16 --inter_threads 16 --data-type $2 --training $3 &
done

}

# bash trn_16c.sh tf2_oppo_model.py bfloat16 1 1024
run_16c $1 $2 $3 $4

  

