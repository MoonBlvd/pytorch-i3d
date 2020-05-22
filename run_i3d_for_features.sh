# mpirun -np 1 python run_i3d_for_features.py \
#        -model_name r2plus1d_18 \
#        -mode rgb \
#        -root /mnt/workspace/datasets/A3D_2.0/frames/ \
#        -val_split A3D_2.0_val.json \
#        -ckpt /mnt/workspace/datasets/A3D_2.0/i3d_outputs/r2plus1d_18/6aw474y3/005000.pt  \
#        -batch_per_gpu 3
mpirun -np 1 python run_i3d_for_features.py \
      -model_name mc3_18 \
      -mode rgb \
      -root /mnt/workspace/datasets/A3D_2.0/frames/ \
      -val_split A3D_2.0_val.json \
      -ckpt /mnt/workspace/datasets/A3D_2.0/i3d_outputs/mc3_18/f5f860vi/003000.pt  \
      -batch_per_gpu 8
#python run_i3d_for_features.py \
#       -model_name r3d_18 \
#       -mode rgb \
#       -root /home/data/vision7/A3D_2.0/frames/ \
#       -val_split A3D_2.0_train.json \
#       -ckpt /home/data/vision7/brianyao/DATA/i3d_outputs/r3d_18/4467dlig/004000.pt  \
#       -batch_per_gpu 6


#mpirun -np 2 
# python run_i3d_for_features.py \
#        -model_name i3d \
#        -mode rgb \
#        -root /home/data/vision7/A3D_2.0/frames/ \
#        -val_split A3D_2.0_val.json \
#        -ckpt /home/data/vision7/brianyao/DATA/i3d_outputs/i3d/tg9s4ff5/004500.pt  \
#        -batch_per_gpu 8
