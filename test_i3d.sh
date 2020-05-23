python test_i3d.py \
    --model_name r2plus1d_18 \
    --mode rgb \
    --checkpoint /u/bryao/work/DATA/i3d_outputs/r2plus1d_18/h0yfcxjg/006500.pt \
    --root /home/data/vision7/A3D_2.0/frames/ \
    --split_file A3D_2.0_val.json \
    --batch_per_gpu 8 \
    --gpu 0 \
    
#r2+1d :  /u/bryao/work/DATA/i3d_outputs/i3d/qnfk82ib/006000.pt \