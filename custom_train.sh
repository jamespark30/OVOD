# python tools/train.py \
#     /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
#     --work-dir /work/ovdet/work_dirs/jh/new_train \
#     # --resume /work/ovdet/checkpoints/iter_90000.pth \
#     # --use_soft_label \
#     # --auto-scale-lr

#     # /work/ovdet/work_dirs/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \


# bash tools/dist_train.sh \
#     2 \
#     /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
#     --work-dir /work/ovdet/work_dirs/merge/jh


# after merge
# bash tools/dist_train.sh \
#     /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
#     2 \
#     --work-dir /work/ovdet/work_dirs/merge/jh \
    # --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing=True \
    # --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_positional_similarity=True \
    # --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_retain_sum=False \
    # --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_alpha=0.004 \
    # --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_groups=3 \
    # --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_permutations=2


# # use multi-proposal
# bash tools/dist_train.sh \
#     /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
#     2 \
#     --work-dir /work/ovdet/work_dirs/merge/jh \
#     --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.use_multi_proposal_bag_weight=1.0 \
#     --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.use_multi_proposal_bag_temp=30.0 \
#     # --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_groups=3 \
#     # --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_permutations=2




# after merge
bash tools/dist_train.sh \
    /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
    2 \
    --work-dir /work/ovdet/work_dirs/final/merge/LS_pos_sim_0.004_0.01 \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing=True \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_positional_similarity=True \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_alpha="[0.004,0.01]" \

# after merge
bash tools/dist_train.sh \
    /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
    2 \
    --work-dir /work/ovdet/work_dirs/final/merge/LS_pos_sim_0.003_0.008 \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing=True \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_positional_similarity=True \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_alpha="[0.003,0.008]" \

# sim_presreving_weight 2500, label_smoothing 0.001
bash tools/dist_train.sh \
    /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
    2 \
    --work-dir /work/ovdet/work_dirs/final/merge/similarity_preserving_45k_2500_w_LS_0.001 \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing=True \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_positional_similarity=False \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.label_smoothing_alpha=0.001 \
    --cfg-options model.roi_head.ovd_cfg.baron_kd.similarity_preserving_weight=2500 \



# 실험 순서
# 1. similarity_preserving_weight hyperparameter tuning (45k)
# 2. 