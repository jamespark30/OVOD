
# CUDA_VISIBLE_DEVICES=0 python tools/train.py \
#     /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
#     --work-dir /work/ovdet/work_dirs/single/jh


# CUDA_VISIBLE_DEVICES=0 python tools/train.py \
#     /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
#     --work-dir /work/ovdet/work_dirs/single/jh \
#     --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing=True \
#     --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_positional_similarity=True \
#     --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_retain_sum=False \
#     --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_alpha=0.001 \

# CUDA_VISIBLE_DEVICES=1 python tools/train.py \
#     /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
#     --work-dir /work/ovdet/work_dirs/single/jh 

CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
    --work-dir /work/ovdet/work_dirs/single/jh \
    --resume /work/ovdet/checkpoints/iter_45000_sim_preserv_20241117_164003.pth \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.similarity_preserving_weight=2000 \
