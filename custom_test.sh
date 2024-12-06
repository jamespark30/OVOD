# GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=12 bash tools/slurm_test.sh PARTITION test \ 
# /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py /work/ovdet/checkpoints/res50_fpn_soco_star_400.pth

# GPUS=16 GPUS_PER_NODE=8 CPUS_PER_TASK=12 bash tools/slurm_train.sh PARTITION train \ 
# configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
# /work/ovdet/checkpoints/res50_fpn_soco_star_400.pth

# GPUS=1 GPUS_PER_NODE=8 CPUS_PER_TASK=12 bash tools/slurm_test.sh PARTITION test \ 
# /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py /work/ovdet/checkpoints/iter_90000.pth


# python -u tools/test.py 
# /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py /work/ovdet/checkpoints/iter_90000.pth



# CUDA_VISIBLE_DEVICES=1 python tools/test.py \
#     /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
#     /work/ovdet/checkpoints/iter_45000.pth \
#     --work-dir /work/ovdet/work_dirs/test/single/jh \
#     --show \
#     --show-dir /work/ovdet/show_dir/ori45k \
#     --cfg-options model.roi_head.ovd_cfg.baron_kd.similarity_preserving_weight=2000 \


CUDA_VISIBLE_DEVICES=1 python tools/test.py \
    /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
    /work/ovdet/checkpoints/iter_45000_sim_preserv_20241117_164003.pth \
    --work-dir /work/ovdet/work_dirs/test/single/jh \
    --show \
    --show-dir /work/ovdet/show_dir/sp45k \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.similarity_preserving_weight=2000 \
