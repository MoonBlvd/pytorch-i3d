#python train_i3d.py \
#mpirun -np 4 
python train_i3d.py \
    -model_name i3d \
    -mode rgb \
    -save_model /home/data/vision7/brianyao/DATA/i3d_outputs/ \
    -root /home/data/vision7/A3D_2.0/frames/ \
    -train_split A3D_2.0_train.json \
    -val_split A3D_2.0_val.json \
    -use_wandb True 
