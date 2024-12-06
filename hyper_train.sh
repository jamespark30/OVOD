# hyperparameter 이것저것 돌려보기


bash tools/dist_train.sh \
            /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
            2 \
            --work-dir /work/ovdet/work_dirs/merge/jh/baseline_45k \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_groups=2 \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_permutations=2


bash tools/dist_train.sh \
            /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
            2 \
            --work-dir /work/ovdet/work_dirs/merge/jh/baseline_45k \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_groups=2 \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_permutations=3

bash tools/dist_train.sh \
            /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
            2 \
            --work-dir /work/ovdet/work_dirs/merge/jh/baseline_45k \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_groups=2 \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_permutations=4

bash tools/dist_train.sh \
            /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
            2 \
            --work-dir /work/ovdet/work_dirs/merge/jh/baseline_45k \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_groups=2 \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_permutations=5


bash tools/dist_train.sh \
            /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
            2 \
            --work-dir /work/ovdet/work_dirs/merge/jh/baseline_45k \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_groups=3 \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_permutations=1


bash tools/dist_train.sh \
            /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
            2 \
            --work-dir /work/ovdet/work_dirs/merge/jh/baseline_45k \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_groups=4 \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_permutations=1


bash tools/dist_train.sh \
            /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
            2 \
            --work-dir /work/ovdet/work_dirs/merge/jh/baseline_45k \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_groups=5 \
            --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_permutations=1          


# max_groups=(5 6)
# max_permutations=(2 3 4)

# for max_group in "${max_groups[@]}"; do
#     for max_permutation in "${max_permutations[@]}"; do
#         bash tools/dist_train.sh \
#             /work/ovdet/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py \
#             2 \
#             --work-dir /work/ovdet/work_dirs/merge/jh/baseline_45k \
#             --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_groups=$max_group \
#             --cfg-options model.roi_head.ovd_cfg.baron_kd.sampling_cfg.max_permutations=$max_permutation
#     done
# done

