#!/bin/bash -l

# If desired, a GPU can be used when making predictions e.g., GPU with index 0.
# However, inference is very fast so a GPU is not necessary.
# To only use a CPU, simply omit the --gpu flag below
gpu=0

# define the path to your chemprop installation
CHEMPROP=/SAMPLE/PATH/chemprop
# export the path as a variable to be used later
export PYTHONPATH=$CHEMPROP:$PYTHONPATH

# csv file containing reaction SMILES to be input to the models
test_path=example_reactions.csv

# number of cpus used to parallelize when loading in the data
num_workers=2

# filename used to save predictions
preds_path=dmpnn_predictions.csv.csv

# use all models to make the prediction
# chemprop's predict script will automatically calculate the mean +- 1 std across the 5 folds
checkpoint_dir=rxn_random
# checkpoint_dir=kmeans

# batch size used during inference
batch_size=32

# activate chemprop environment
source activate chemprop
which python
python -c "import torch;print(torch.cuda.device_count());print(torch.cuda.is_available())"

# -u: Force stdin, stdout and stderr to be totally unbuffered.
# On systems where it matters, also put stdin, stdout and stderr in binary mode
python -u $CHEMPROP/predict.py \
--gpu $gpu \
--num_workers $num_workers \
--test_path $test_path \
--checkpoint_dir $checkpoint_dir \
--preds_path $preds_path \
--batch_size $batch_size \
--ensemble_variance \
--no_features_scaling

