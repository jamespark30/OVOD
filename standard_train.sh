# 실험용
# python tools/train.py \
#     /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
#     --work-dir /work/ovdet/work_dirs/jh/standard_train \
#     --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing=True \
#     --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_positional_similarity=True \
#     --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_retain_sum=False \
#     --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_alpha=0.005 \


bash tools/dist_train.sh \
    /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
    2 \
    --work-dir /work/ovdet/work_dirs/jh/standard_train \
    --resume /work/ovdet/checkpoints/iter_90000.pth \