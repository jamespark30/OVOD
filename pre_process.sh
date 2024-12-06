python tools/pre_processors/keep_coco_base.py \
      --json_path data/coco/annotations/instances_train2017.json \
      --out_path data/coco/wusize/instances_train2017_base.json

python tools/pre_processors/keep_coco_base.py \
      --json_path data/coco/annotations/instances_val2017.json \
      --out_path data/coco/wusize/instances_val2017_base.json

python tools/pre_processors/keep_coco_novel.py \
      --json_path data/coco/annotations/instances_val2017.json \
      --out_path data/coco/wusize/instances_val2017_novel.json