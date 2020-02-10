python run_i3d_for_features.py \
       -model_name r3d_18 \
       -mode rgb \
       -root /home/data/vision7/A3D_2.0/frames/ \
       -val_split A3D_2.0_train.json \
       -ckpt /home/data/vision7/brianyao/DATA/i3d_outputs/r3d_18/4467dlig/004000.pt  \
       -batch_per_gpu 6

# python run_i3d_for_features.py \
#        -model_name c3d \
#        -mode rgb \
#        -root /home/data/vision7/A3D_2.0/frames/ \
#        -val_split A3D_2.0_train.json \
#        -ckpt /home/data/vision7/brianyao/DATA/i3d_outputs/c3d/8w8xilwx/002500.pt  \
#        -batch_per_gpu 8
       
# mpirun -np 2 
# python run_i3d_for_features.py \
#        -model_name i3d \
#        -mode rgb \
#        -root /home/data/vision7/A3D_2.0/frames/ \
#        -val_split A3D_2.0_train.json \
#        -ckpt /home/data/vision7/brianyao/DATA/i3d_outputs/i3d/tg9s4ff5/004500.pt  \
#        -batch_per_gpu 8