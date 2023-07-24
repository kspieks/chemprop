#!/bin/bash -l

gpu=0

CHEMPROP=/SAMPLE/PATH/chemprop
PRETRAIN=/PATH/TO/PRETRAINED/MODEL
DATA=/PATH/TO/DATA
export PYTHONPATH=$CHEMPROP:$PYTHONPATH

### TrainArgs ###
# general args
data_path=$DATA/fwd_rev_rxns.csv
checkpoint_path=$PRETRAIN/model.pt
dataset_type=regression
target_columns="DeltaG_ET_gas_molar_kcal_mol DeltaG_EP_gas_molar_kcal_mol"

metric=rmse
extra_metrics="mae r2" 	# additional evaluation metrics. not used for early stopping

save_dir=model
cache_cutoff=12000	# max number of molecules in dataset to allow caching

epochs=55
batch_size=50
final_lr=1e-6
init_lr=1e-5
max_lr=1e-4
warmup_epochs=5.0
grad_clip=10

log_frequency=1 	# number of batches between each logging of the training loss

split_type=index_predetermined
crossval_index_file=$DATA/kmeans_splits.pkl

# model arguments
hidden_size=300 	# dimensionality of hidden layers in mpn
depth=3				# number of message passing steps
dropout=0
activation=LeakyReLU
ffn_num_layers=2 	# number of layers in FFN after MPN encoding

aggregation=sum

reaction_mode=reac_diff

echo "Start time: $(date '+%Y-%m-%d_%H:%M:%S')"
source activate chemprop
which python
python -c "import torch;print(torch.cuda.device_count());print(torch.cuda.is_available())"

# -u: Force stdin, stdout and stderr to be totally unbuffered. On systems where it matters, also put stdin, stdout and stderr in binary mode
python -u $CHEMPROP/train.py \
--gpu $gpu \
--data_path $data_path \
--num_workers 2 \
--checkpoint_path $checkpoint_path \
--dataset_type $dataset_type \
--save_dir $save_dir \
--target_columns $target_columns \
--save_preds \
--split_type $split_type \
--crossval_index_file $crossval_index_file \
--log_frequency $log_frequency \
--metric $metric \
--extra_metrics $extra_metrics \
--cache_cutoff $cache_cutoff \
--reaction \
--reaction_mode $reaction_mode \
--explicit_h \
--aggregation $aggregation \
--depth $depth \
--ffn_num_layers $ffn_num_layers \
--hidden_size $hidden_size \
--activation $activation \
--dropout $dropout \
--epochs $epochs \
--batch_size $batch_size \
--init_lr $init_lr \
--max_lr $max_lr \
--final_lr $final_lr \
--grad_clip $grad_clip \
--warmup_epochs $warmup_epochs \
--no_features_scaling \
--show_individual_scores \
--validate_on_first_target

echo "End time: $(date '+%Y-%m-%d_%H:%M:%S')"
