# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.models.roi_heads import StandardRoIHead
from mmengine.structures import InstanceData
from ovdet.methods.builder import OVD


################################################
import os
import sys
sys.path.append("/work/ovdet")

import torch
import mmcv
from mmengine.visualization import Visualizer
################################################


@MODELS.register_module()
class OVDStandardRoIHead(StandardRoIHead):
    def __init__(self, clip_cfg=None, ovd_cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if clip_cfg is None:
            self.clip = None
        else:
            self.clip = MODELS.build(clip_cfg)
        if ovd_cfg is not None:
            for k, v in ovd_cfg.items():
                # self.register_module(k, OVD.build(v))   # not supported in pt1.8.1
                setattr(self, k, OVD.build(v))

    def _bbox_forward(self, x, rois):
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats, self.clip)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results


    def visualize(self, x, sampling_results_list, *args, **kwargs):
        
        if isinstance(sampling_results_list[0], InstanceData):
            print("start-------------------------")
            for res in sampling_results_list:
                print(res)
                bbox_count = 0
                print("normed_boxes.shape : ", len(res.normed_boxes))
                num_spanned_boxes = [len(proposal) for proposal in res.normed_boxes]
                print(num_spanned_boxes)
                
                save_dir = os.path.basename(res.img_path)
                save_dir = os.path.join('/work/ovdet/work_dirs/jh/new_outputs', save_dir)
                
                for i in range(len(res.normed_boxes)):  # each proposal
                    box_id_count = 0
                    for j in range(len(res.normed_boxes[i])): # each spanned box
                        for k in range(len(res.normed_boxes[i][j])): # each permutation
                            print(i, "th proposal ",
                                  j, "th spanned box ",
                                  k, "th permutation ",
                                  "box_ids", res.box_ids[i][box_id_count])
                            
                            image = mmcv.imread(res.img_path, channel_order='rgb')
                            visualizer = Visualizer(image=image,
                                                    vis_backends=[dict(type='LocalVisBackend')],
                                                    save_dir=save_dir)
                            for _ in range(res.normed_boxes[i][j][k].shape[0]): # 각 candiate box
                                # print("bbox coordinates : ", res.bboxes[bbox_count])
                                scaled_bbox = res.bboxes[bbox_count].clone()
                                scaled_bbox[[0, 2]] /= res.scale_factor[0]
                                scaled_bbox[[1, 3]] /= res.scale_factor[1]
                                visualizer.draw_bboxes(scaled_bbox, edge_colors='g', line_widths=3)
                                bbox_count = bbox_count + 1
                            
                            scaled_s_bbox = res.spanned_boxes[i][j].clone()
                            scaled_s_bbox[[0, 2]] /= res.scale_factor[0]
                            scaled_s_bbox[[1, 3]] /= res.scale_factor[1]
                            visualizer.draw_bboxes(scaled_s_bbox, edge_colors='r', line_widths=3, line_styles='--')
                            
                            visualizer.add_image('test', visualizer.get_image(), step=((i+1)*100+(box_id_count+1)))
                            box_id_count = box_id_count + 1
            print("final bbox count : ", bbox_count)
        
        # if isinstance(sampling_results_list[0], InstanceData):
        #     print("-------------------------")
        #     for res in sampling_results_list:
        #         print(res)
        #         for (i, _) in enumerate(res.box_ids): # each proposal
        #             for (j, _) in enumerate(res.box_ids[i]):
        #             image = mmcv.imread(res.img_path, channel_order='rgb')
        #             visualizer = Visualizer(image=image,
        #                                     vis_backends=[dict(type='LocalVisBackend')],
        #                                     save_dir='/work/ovdet/work_dirs/jh/outputs')
                    
        #             print("spanned_boxes : []", res.spanned_boxes[i])
        #             # print(res.normed_boxes[i])
                    
        #             visualizer.draw_bboxes(torch.cat([box.unsqueeze(0) for box in res.spanned_boxes[i]], dim=1), edge_colors='r', line_widths=3)
        #             # visualizer.draw_bboxes(torch.cat([box.unsqueeze(0) for box in res.normed_boxes[i]], dim=1), edge_colors='g', line_widths=3)
                    
        #             visualizer.add_image('test', visualizer.get_image())
        else:
            print('length of sampling_results_list : ', len(sampling_results_list))
        
        print("end-------------------------")
            
        # bbox_feats = self.bbox_roi_extractor(
        #         x[:self.bbox_roi_extractor.num_inputs], rois)
        
        # https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/en/_static/image/cat_and_dog.png
        
        # # single bbox formatted as [xyxy]
        # visualizer.draw_bboxes(torch.tensor([72, 13, 179, 147]))
        
        # draw multiple bboxes
        # visualizer.set_image(image=image)
        # visualizer.draw_bboxes(torch.tensor([[33, 120, 209, 220], [72, 13, 179, 147]]))
        # visualizer.draw_bboxes(torch.tensor([72, 13, 179, 147]),
        #                 edge_colors='r',
        #                 line_widths=3)
        # visualizer.draw_bboxes(torch.tensor([[33, 120, 209, 220]]),line_styles='--')
        
        # visualizer.show()
        
        
    def run_ovd(self, x, batch_data_samples, rpn_results_list, ovd_name, batch_inputs,
                *args, **kwargs):
        ovd_method = getattr(self, ovd_name)

        sampling_results_list = list(map(ovd_method.sample, rpn_results_list, batch_data_samples))
        if isinstance(sampling_results_list[0], InstanceData):
            rois = bbox2roi([res.bboxes for res in sampling_results_list])
        else:
            sampling_results_list_ = []
            bboxes = []
            for sampling_results in sampling_results_list:
                bboxes.append(torch.cat([res.bboxes for res in sampling_results]))
                sampling_results_list_ += sampling_results
            rois = bbox2roi(bboxes)
            sampling_results_list = sampling_results_list_

        #########################################################################################################
        # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        # print("length of sampling_results_list : ", len(sampling_results_list))
        
        # self.visualize(x, sampling_results_list)
        
        # print(sampling_results_list[0])
        # roi : torch.Size([14, 5])
        # bbox_feats.shape : torch.Size([14, 256, 7, 7])
        # print('len(x): ',len(x))
        # print('x[0]: ', x[0].shape)
        # x[0] : torch.Size([1, 256, 192, 256])
        
        
        # print(ddddd)
        #########################################################################################################
        
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self.bbox_head.vision_to_language(bbox_feats)
        # For baron, region embeddings are pseudo words
        
        # print("region_embeddings.shape : ", region_embeddings.shape)
        
        return ovd_method.get_losses(region_embeddings, sampling_results_list, self.clip, batch_inputs)
