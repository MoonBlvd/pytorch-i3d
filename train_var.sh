# python train_i3d.py \
#     -model_name i3d \
#     -mode rgb \
#     -save_model /home/data/vision7/brianyao/DATA/i3d_outputs/ \
#     -root /home/data/vision7/A3D_2.0/frames/ \
#     -train_split A3D_2.0_train.json \
#     -val_split A3D_2.0_val.json \
#     -batch_per_gpu 16 \
#     -gpu 0 \
#     -checkpoint_peroid 500 \
#     -use_wandb

# python train_i3d.py \
#     -model_name c3d \
#     -mode rgb \
#     -save_model /home/data/vision7/brianyao/DATA/i3d_outputs/ \
#     -root /home/data/vision7/A3D_2.0/frames/ \
#     -train_split A3D_2.0_train.json \
#     -val_split A3D_2.0_val.json \
#     -batch_per_gpu 16 \
#     -gpu 0 \
#     -checkpoint_peroid 500 \
#     -use_wandb

python train_i3d.py \
    -model_name r2plus1d_18 \
    -mode rgb \
    -save_model /nfs/tynamo/home/data/vision7/brianyao/DATA/i3d_outputs/ \
    -root /nfs/tynamo/home/data/vision7/A3D_2.0/frames/ \
    -train_split A3D_2.0_train.json \
    -val_split A3D_2.0_val.json \
    -batch_per_gpu 16 \
    -gpu 0 \
    -checkpoint_peroid 500 \
    -use_wandb

python train_i3d.py \
    -model_name mc3_18 \
    -mode rgb \
    -save_model /nfs/tynamo/home/data/vision7/brianyao/DATA/i3d_outputs/ \
    -root /nfs/tynamo/home/data/vision7/A3D_2.0/frames/ \
    -train_split A3D_2.0_train.json \
    -val_split A3D_2.0_val.json \
    -batch_per_gpu 16 \
    -gpu 0 \
    -checkpoint_peroid 500 \
    -use_wandb

python train_i3d.py \
    -model_name r3d_18 \
    -mode rgb \
    -save_model /nfs/tynamo/home/data/vision7/brianyao/DATA/i3d_outputs/ \
    -root /nfs/tynamo/home/data/vision7/A3D_2.0/frames/ \
    -train_split A3D_2.0_train.json \
    -val_split A3D_2.0_val.json \
    -batch_per_gpu 16 \
    -gpu 0 \
    -checkpoint_peroid 500 \
    -use_wandb