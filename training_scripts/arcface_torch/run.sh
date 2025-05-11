# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train_v2.py $@
source $HOME/mlops_env/bin/activate
python3 train_v2.py configs/ms1mv3_r50_onegpu
# torchrun \
#   --nproc_per_node=2 \
#   --rdzv_backend=c10d \
#   --rdzv_endpoint=127.0.0.1:12584 \
#   train_v2.py configs/ms1mv3_r50 
deactivate