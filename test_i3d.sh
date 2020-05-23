python test_i3d.py \
    --model_name i3d \
    --mode rgb \
    --checkpoint /u/bryao/work/DATA/i3d_outputs/i3d/qnfk82ib/006000.pt \
    --root /home/data/vision7/A3D_2.0/frames/ \
    --split_file A3D_2.0_val.json \
    --batch_per_gpu 12 \
    --gpu 0 

python test_i3d.py \
    --model_name c3d \
    --mode rgb \
    --checkpoint /u/bryao/work/DATA/i3d_outputs/c3d/4w57dd32/005500.pt \
    --root /home/data/vision7/A3D_2.0/frames/ \
    --split_file A3D_2.0_val.json \
    --batch_per_gpu 12 \
    --gpu 0 \

# r2plus1d_18: /u/bryao/work/DATA/i3d_outputs/r2plus1d_18/h0yfcxjg/007000.pt
# i3d: /u/bryao/work/DATA/i3d_outputs/i3d/qnfk82ib/006000.pt
# c3d: /u/bryao/work/DATA/i3d_outputs/c3d/4w57dd32/005500.pt
# 