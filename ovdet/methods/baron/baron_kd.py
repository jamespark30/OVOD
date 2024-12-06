import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import roi_align
from mmengine.structures import InstanceData
from mmengine.runner.amp import autocast
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh, bbox2roi
from mmdet.registry import MODELS
from ovdet.methods.builder import OVD
from ovdet.utils import multi_apply
from .baron_base import BaronBase
from .utils import repeat_crops_and_get_att_mask
from .neighborhood_sampling import NeighborhoodSampling
from .boxes_cache import BoxesCache

##############################################
# import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image

import mmcv
from mmengine.visualization import Visualizer

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from datetime import datetime
##############################################

def visualize_two_matrices(student_matrix, teacher_matrix, file_name, box_coords, image_box_coords):
    student_matrix = student_matrix.detach().cpu().numpy()
    teacher_matrix = teacher_matrix.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    norm_student = mcolors.Normalize(vmin=student_matrix.min(), vmax=student_matrix.max())
    norm_teacher = mcolors.Normalize(vmin=teacher_matrix.min(), vmax=teacher_matrix.max())    
    
    axes[0].imshow(student_matrix, cmap='viridis', norm=norm_student, interpolation='none')
    axes[0].set_title("Student")
    # fig.colorbar(axes[0].images[0], ax=axes[0], orientation='vertical')
    
    axes[1].imshow(teacher_matrix, cmap='viridis', norm=norm_teacher, interpolation='none')
    axes[1].set_title("Teacher")
    # fig.colorbar(axes[1].images[0], ax=axes[1], orientation='vertical')
    
    fig.colorbar(axes[0].images[0], ax=axes, orientation='vertical')
    
    if box_coords:
        for (x, y, width, height) in box_coords:
            rect1 = patches.Rectangle((x-0.5, y-0.5), width, height, linewidth=0.5, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect1)
            
            rect2 = patches.Rectangle((x-0.5, y-0.5), width, height, linewidth=0.5, edgecolor='red', facecolor='none')
            axes[1].add_patch(rect2)
        for (x, y, width, height) in image_box_coords:
            rect3 = patches.Rectangle((x-0.5, y-0.5), width, height, linewidth=1, edgecolor='blue', facecolor='none')
            axes[0].add_patch(rect3)
            
            rect4 = patches.Rectangle((x-0.5, y-0.5), width, height, linewidth=1, edgecolor='blue', facecolor='none')
            axes[1].add_patch(rect4)

    file_name = "/work/ovdet/work_dirs/jh/aff_matrices_sp45k/" + file_name + ".png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close()
    
def matrix_logarithm(input_tensor):
    # Half precision일 경우 float32로 변환
    if input_tensor.dtype == torch.float16:
        input_tensor = input_tensor.to(torch.float32)
    
    # Hermitian이 아니거나 정방행렬이 아닐 경우 에러 처리
    if input_tensor.size(0) != input_tensor.size(1):
        raise ValueError("Matrix logarithm is only defined for square matrices.")
    
    # 고유값 분해 수행 (복소수 고유값 포함)
    eigenvalues, eigenvectors = torch.linalg.eig(input_tensor)

    # 고유값에 대한 NaN 및 음수 체크 후 처리
    if torch.any(eigenvalues.real <= 0):
        print("Warning: Matrix contains non-positive eigenvalues.")
    
    # 고유값의 로그 계산 (복소수일 경우 복소수 로그 포함)
    log_eigenvalues = torch.diag(torch.log(eigenvalues))

    # 행렬 로그 재구성: V * log(D) * V^(-1)
    matrix_log = eigenvectors @ log_eigenvalues @ torch.linalg.inv(eigenvectors)

    # 복소수 결과에서 실수만 필요한 경우 실수 부분만 반환
    return matrix_log.real if torch.is_complex(matrix_log) else matrix_log

# def matrix_logarithm(input_tensor):
#     # Half precision일 경우 float32로 변환
#     if input_tensor.dtype == torch.float16:
#         input_tensor = input_tensor.to(torch.float32)
    
#     # Hermitian이 아니거나 정방행렬이 아닐 경우 에러 처리
#     if input_tensor.size(0) != input_tensor.size(1):
#         raise ValueError("Matrix logarithm is only defined for square matrices.")
    
#     # # 고유값 분해 수행 (Hermitian 행렬이 아닌 경우 복소수 고유값 분해)
#     # eigenvalues, eigenvectors = torch.linalg.eig(input_tensor)

#     # 고유값 분해 수행 (Hermitian 행렬 여부에 따라 적절한 분해 함수 선택)
#     if torch.allclose(input_tensor, input_tensor.T.conj()):  # Hermitian 여부 확인
#         eigenvalues, eigenvectors = torch.linalg.eigh(input_tensor)
#     else:
#         eigenvalues, eigenvectors = torch.linalg.eig(input_tensor)

#     # 고유값 중 0 이하의 값은 로그에서 제외하고 0으로 설정
#     log_eigenvalues = torch.diag(torch.where(eigenvalues > 0, torch.log(eigenvalues), torch.zeros_like(eigenvalues)))

#     # 행렬 로그 재구성: V * log(D) * V^(-ççç1)
#     matrix_log = eigenvectors @ log_eigenvalues @ torch.linalg.inv(eigenvectors)

#     # NaN 값 체크 및 대체 (NaN을 0으로 대체)
#     matrix_log = torch.where(torch.isnan(matrix_log), torch.zeros_like(matrix_log), matrix_log)

#     # 디버깅 정보 출력
#     #print("Eigenvalues:", eigenvalues)
#     #print("Log of Eigenvalues:", torch.log(eigenvalues))
#     #print("Log Eigenvalues Diagonal Matrix:", log_eigenvalues)
#     #print("Matrix Logarithm Result:", matrix_log)
    
#     return matrix_log.real if torch.is_complex(matrix_log) else matrix_log

# def matrix_logarithm(input_tensor, eps=1e-6):
#     # Half precision일 경우 float32로 변환
#     if input_tensor.dtype == torch.float16:
#         input_tensor = input_tensor.to(torch.float32)

#     # 입력 텐서 정규화
#     input_tensor = F.normalize(input_tensor, p='fro', dim=(-2, -1))

#     # 수치적 안정성을 위해 작은 값 추가
#     input_tensor = input_tensor + eps * torch.eye(
#         input_tensor.size(0), 
#         device=input_tensor.device, 
#         dtype=input_tensor.dtype
#     )
    
#     # # 정방행렬 체크
#     # if input_tensor.size(0) != input_tensor.size(1):
#     #     raise ValueError("Matrix logarithm is only defined for square matrices.")
    
#     # # Hermitian 여부 확인 및 적절한 고유값 분해 수행
#     # if torch.allclose(input_tensor, input_tensor.T.conj(), rtol=1e-5, atol=1e-8):
#     #     eigenvalues, eigenvectors = torch.linalg.eigh(input_tensor)
#     #     # eigh는 항상 실수 고유값을 반환하므로 추가 처리가 필요 없음
#     #     log_eigenvalues = torch.diag(torch.log(torch.clamp(eigenvalues, min=eps)))
#     # else:
#     #     eigenvalues, eigenvectors = torch.linalg.eig(input_tensor)
#     #     # 복소수 고유값에 대해 실수부만 사용
#     #     if torch.is_complex(eigenvalues):
#     #         eigenvalues = eigenvalues.real
#     #     log_eigenvalues = torch.diag(torch.log(torch.clamp(eigenvalues, min=eps)))

#     # 정방행렬 체크
#     if input_tensor.size(0) != input_tensor.size(1):
#         raise ValueError("Matrix logarithm is only defined for square matrices.")
    
#     # Hermitian 여부 확인 및 적절한 고유값 분해 수행
#     if torch.allclose(input_tensor, input_tensor.T.conj(), rtol=1e-5, atol=1e-8):
#         eigenvalues, eigenvectors = torch.linalg.eigh(input_tensor)
#     else:
#         eigenvalues, eigenvectors = torch.linalg.eig(input_tensor)
#         if torch.is_complex(eigenvalues):
#             eigenvalues = eigenvalues.real
    
#     # 고유값의 로그 계산 시 수치적 안정성 보장
#     log_eigenvalues = torch.diag(torch.log(torch.clamp(eigenvalues, min=eps, max=1e3)))
    
#     # 행렬 로그 재구성
#     matrix_log = eigenvectors @ log_eigenvalues @ torch.linalg.inv(eigenvectors)
    
#     # 복소수 결과가 나온 경우 실수부만 추출
#     if torch.is_complex(matrix_log):
#         matrix_log = matrix_log.real
    
#     # NaN/Inf 처리
#     matrix_log = torch.where(
#         torch.isnan(matrix_log) | torch.isinf(matrix_log),
#         torch.zeros_like(matrix_log),
#         matrix_log
#     )

#     # 출력 값 범위 제한
#     matrix_log = torch.clamp(matrix_log, min=-1e3, max=1e3)
#     return matrix_log
    
def process_sampling_result_per_image(sampling_result, device):
    # add region dropout
    spanned_boxes = sampling_result.spanned_boxes
    normed_boxes = sampling_result.normed_boxes
    box_ids = sampling_result.box_ids
    seq_ids = [list(map(box_ids2seq_id, box_ids_)) for box_ids_ in box_ids]
    seq_ids_per_image = []
    start_id = 0
    for seq_ids_ in seq_ids:
        seq_ids_per_image.extend([box_id + start_id for box_id in seq_ids_])
        start_id += (max(seq_ids_) + 1)
    sampling_result.set_field(name='seq_ids', value=seq_ids_per_image,
                              field_type='metainfo', dtype=None)

    group_split = [len(grp) * grp[0].shape[0] for ori in normed_boxes for grp in ori]
    origin_split = [sum([len(grp) * grp[0].shape[0] for grp in ori]) for ori in normed_boxes]
    perms_split = [perm.shape[0] for ori in normed_boxes for grp in ori for perm in grp]

    seq_level_origin_split = [sum([len(grp) for grp in ori]) for ori in normed_boxes]
    seq_level_group_split = [len(grp) for ori in normed_boxes for grp in ori]
    
    normed_boxes = torch.cat([torch.cat(grp, dim=0)
                              for ori in normed_boxes for grp in ori], dim=0).to(device)
    spanned_boxes = torch.cat([torch.stack(ori, dim=0) for ori in spanned_boxes]).to(device)

    return normed_boxes, spanned_boxes, origin_split, group_split, perms_split, \
           seq_level_origin_split, seq_level_group_split


def box_ids2seq_id(box_ids):
    box_ids_copy = box_ids.copy()
    box_ids_sorted = sorted(box_ids_copy, reverse=True)
    box_ids_str = ''.join([str(box_id) for box_id in box_ids_sorted])

    return int(box_ids_str)


@OVD.register_module()
class BaronKD(BaronBase):
    def __init__(self, bag_weight, single_weight, use_attn_mask, bag_temp, single_temp,
                 use_gt,
                 clip_data_preprocessor,
                 boxes_cache=None,
                 label_smoothing=False,
                 label_smoothing_positional_similarity=False,
                 label_smoothing_retain_sum=False,
                 label_smoothing_alpha=0.1,
                 multi_proposal_bag_weight=0.0,
                 multi_proposal_bag_temp=30.0,
                 similarity_preserving_weight=0.0,
                 **kwargs):
        super(BaronKD, self).__init__(**kwargs)
        self.neighborhood_sampling = NeighborhoodSampling(**self.sampling_cfg)
        self.bag_temp = bag_temp        # 30.0
        self.single_temp = single_temp  # 50.0
        self.use_attn_mask = use_attn_mask
        self.bag_weight = bag_weight
        self.single_weight = single_weight
        self.use_gt = use_gt
        self.clip_data_preprocessor = MODELS.build(clip_data_preprocessor)
        if boxes_cache is not None:
            boxes_cache.update(num_proposals=self.sampling_cfg['topk'],
                               nms_thr=self.sampling_cfg['nms_thr'],
                               score_thr=self.sampling_cfg['objectness_thr'])
            self.boxes_cache = BoxesCache(**boxes_cache)
        else:
            self.boxes_cache = None
        ##################################################################
        self.label_smoothing = label_smoothing,
        self.label_smoothing_positional_similarity = label_smoothing_positional_similarity,
        self.label_smoothing_retain_sum = label_smoothing_retain_sum,
        self.label_smoothing_alpha = label_smoothing_alpha,
        self.multi_proposal_bag_weight = multi_proposal_bag_weight
        self.multi_proposal_bag_temp = multi_proposal_bag_temp
        self.similarity_preserving_weight = similarity_preserving_weight
        ##################################################################

    def visualize(self, topk_proposals, nmsed_proposals, *args, **kwargs):
        # print("start---------------------------------------------------------------------------")
        # print(len(topk_proposals), len(nmsed_proposals))
        
        img_count = 0
        if isinstance(nmsed_proposals[0], InstanceData):
            for i, res in enumerate(nmsed_proposals):
                save_dir = os.path.basename(res.img_path)
                save_dir = os.path.join('/work/ovdet/work_dirs/jh/standard_outputs2', save_dir)
                
                if i==0:
                    image = mmcv.imread(res.img_path, channel_order='rgb')
                    visualizer = Visualizer(image=image,
                                            vis_backends=[dict(type='LocalVisBackend')],
                                            save_dir=save_dir)
                elif save_dir!=last_save_dir:
                    print(i)
                    visualizer.add_image('test', visualizer.get_image(), step=img_count)
                    img_count = img_count + 1
                    
                    image = mmcv.imread(res.img_path, channel_order='rgb')
                    visualizer = Visualizer(image=image,
                                            vis_backends=[dict(type='LocalVisBackend')],
                                            save_dir=save_dir)
                
                scaled_bbox = res.bboxes.clone()
                scaled_bbox[:, [0, 2]] /= res.scale_factor[0]
                scaled_bbox[:, [1, 3]] /= res.scale_factor[1]
                visualizer.draw_bboxes(scaled_bbox, edge_colors='g', line_widths=2)
                    
                last_save_dir = save_dir
            visualizer.add_image('test', visualizer.get_image(), step=img_count)
            img_count = img_count + 1
        
        
        # if isinstance(sampling_results_list[0], InstanceData):
        #     for res in sampling_results_list:
        #         print(res)
        #         bbox_count = 0
        #         print("normed_boxes.shape : ", len(res.normed_boxes))
        #         num_spanned_boxes = [len(proposal) for proposal in res.normed_boxes]
        #         print(num_spanned_boxes)
                
        #         save_dir = os.path.basename(res.img_path)
        #         save_dir = os.path.join('/work/ovdet/work_dirs/jh/new_outputs', save_dir)
                
        #         for i in range(len(res.normed_boxes)):  # each proposal
        #             box_id_count = 0
        #             for j in range(len(res.normed_boxes[i])): # each spanned box
        #                 for k in range(len(res.normed_boxes[i][j])): # each permutation
        #                     print(i, "th proposal ",
        #                           j, "th spanned box ",
        #                           k, "th permutation ",
        #                           "box_ids", res.box_ids[i][box_id_count])
                            
        #                     image = mmcv.imread(res.img_path, channel_order='rgb')
        #                     visualizer = Visualizer(image=image,
        #                                             vis_backends=[dict(type='LocalVisBackend')],
        #                                             save_dir=save_dir)
        #                     for _ in range(res.normed_boxes[i][j][k].shape[0]): # 각 candiate box
        #                         # print("bbox coordinates : ", res.bboxes[bbox_count])
        #                         scaled_bbox = res.bboxes[bbox_count].clone()
        #                         scaled_bbox[[0, 2]] /= res.scale_factor[0]
        #                         scaled_bbox[[1, 3]] /= res.scale_factor[1]
        #                         visualizer.draw_bboxes(scaled_bbox, edge_colors='g', line_widths=3)
        #                         bbox_count = bbox_count + 1
                            
        #                     scaled_s_bbox = res.spanned_boxes[i][j].clone()
        #                     scaled_s_bbox[[0, 2]] /= res.scale_factor[0]
        #                     scaled_s_bbox[[1, 3]] /= res.scale_factor[1]
        #                     visualizer.draw_bboxes(scaled_s_bbox, edge_colors='r', line_widths=3, line_styles='--')
                            
        #                     visualizer.add_image('test', visualizer.get_image(), step=((i+1)*100+(box_id_count+1)))
        #                     box_id_count = box_id_count + 1
        #     print("final bbox count : ", bbox_count)
    
        else:
            print('length of sampling_results_list : ', len(nmsed_proposals))
        
        # print("end---------------------------------------------------------------------------")
        
    def _sample_on_topk(self, topk_proposals):
        img_shape = topk_proposals.img_shape
        h, w = img_shape
        device = topk_proposals.scores.device
        image_box = torch.tensor([0.0, 0.0, w - 1.0, h - 1.0], device=device)

        if len(topk_proposals) == 0:
            topk_proposals = InstanceData(bboxes=image_box[None],
                                          scores=torch.tensor([1.0], dtype=device),
                                          metainfo=topk_proposals.metainfo.copy())

        nmsed_proposals = self.preprocess_proposals(topk_proposals,
                                                    image_box[None],
                                                    self.sampling_cfg['shape_ratio_thr'],
                                                    self.sampling_cfg['area_ratio_thr'],
                                                    self.sampling_cfg['objectness_thr'],
                                                    self.sampling_cfg['nms_thr'])
        if self.boxes_cache is not None:
            nmsed_proposals = self.boxes_cache(nmsed_proposals)
            
        # print(topk_proposals.bboxes.shape, nmsed_proposals.bboxes.shape)
        # self.visualize(topk_proposals, nmsed_proposals)
        # print(ddd)
        
        func = self.neighborhood_sampling.sample
        boxes = nmsed_proposals.bboxes.tolist()
        groups_per_proposal, normed_boxes, spanned_boxes, box_ids = \
            multi_apply(func, boxes,
                        [img_shape] * len(nmsed_proposals))   # can be time-consuming
        
        ########################################################
        # groups_per_proposal : 각 box 좌표 (실제)  -> max_group(G)=3 개의 sampling 된 애들이 구성이 max_permutations=2 수 씩 존재
        # normed_boxes  : 각 box 좌표 (normed)
        # spanned boxes : enclosing box 좌표 (실제)
        # box_ids : [5, 7, 4, 2], [2, 4, 5, 7],
        ########################################################
        
        new_boxes = torch.cat([perm for single_proposal in groups_per_proposal
                               for single_group in single_proposal for perm in single_group], dim=0).to(device)
        metainfo = topk_proposals.metainfo.copy()
        metainfo.update(normed_boxes=normed_boxes,
                        spanned_boxes=spanned_boxes,
                        box_ids=box_ids)
        sampled_instances = InstanceData(bboxes=new_boxes, metainfo=metainfo)

        return sampled_instances

    def _sample_topk_proposals(self, proposals_per_image):
        num = min(len(proposals_per_image), self.sampling_cfg['topk'])
        _, topk_inds = proposals_per_image.scores.topk(num)

        return proposals_per_image[topk_inds]

    @staticmethod
    def _add_gt_boxes(proposals, gt_boxes):
        if len(gt_boxes) == 0:
            return proposals
        proposal_bboxes = proposals.bboxes
        proposal_scores = proposals.scores
        gt_scores = torch.ones_like(gt_boxes[:, 0])

        return InstanceData(bboxes=torch.cat([gt_boxes, proposal_bboxes]),
                            scores=torch.cat([gt_scores, proposal_scores]),
                            metainfo=proposals.metainfo)

    def sample(self, rpn_results, batch_data_sample, **kwargs):
        rpn_results.set_metainfo(batch_data_sample.metainfo)
        topk_proposals = self._sample_topk_proposals(rpn_results)
        if self.use_gt:
            topk_proposals = self._add_gt_boxes(topk_proposals,
                                                batch_data_sample.gt_instances.bboxes)
        sampling_result = self._sample_on_topk(topk_proposals)

        return sampling_result

    @torch.no_grad()
    def _bbox_clip_image(self, spanned_boxes, clip_images,
                         seqs_split_by_group,
                         normed_boxes_split_by_perms,
                         clip_model):
        # TODO: repeat and mask
        image_encoder = clip_model.image_encoder
        num_groups_per_image = [b.shape[0] for b in spanned_boxes]
        clip_input_size = image_encoder.input_resolution
        
        #############################################################################
        # save_dir = "/work/ovdet/work_dirs/jh/test_batch_original_img"
        # # unnormalize = transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[2.0, 2.0, 2.0])
        # for i in range(clip_images.shape[0]):
        #     file_path = os.path.join(save_dir, f'image_{i+1}.png')  # 파일 이름 설정
        #     save_image(clip_images[i], file_path)
        # print(f'Saved {file_path}')
        #############################################################################
        
        clip_images = self.clip_data_preprocessor({'inputs': clip_images})['inputs']

        input_to_clip = roi_align(
            clip_images, bbox2roi(spanned_boxes), (clip_input_size, clip_input_size),
            1.0, 2, 'avg', True)
        input_to_clip = input_to_clip.split(num_groups_per_image, dim=0)
        repeated_crops, attn_masks = multi_apply(repeat_crops_and_get_att_mask,
                                                 input_to_clip, seqs_split_by_group,
                                                 normed_boxes_split_by_perms,
                                                 num_heads=image_encoder.num_heads,
                                                 grid_size=image_encoder.attn_resolution,
                                                 use_attn_mask=self.use_attn_mask)

        repeated_crops = torch.cat(repeated_crops, dim=0)
        if attn_masks[0] is None:
            attn_masks = None
        else:
            attn_masks = torch.cat(attn_masks, dim=0)
        
        #############################################################################
        # print("repeated crops: ", repeated_crops.shape)
        # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        
        # save_dir = "/work/ovdet/work_dirs/jh/test_batch_img"
        # for i in range(repeated_crops.shape[0]):
        #     img = repeated_crops[i].cpu()
            
        #     file_path = os.path.join(save_dir, f'image_{i+1}.png')  # 파일 이름 설정
            
        #     save_image(img, file_path)
        #     print(f'Saved {file_path}')
        #############################################################################
            
        clip_img_features, clip_img_tokens = image_encoder.encode_image(
            repeated_crops, normalize=True, return_tokens=True, attn_masks=attn_masks)
        # print(f'clip_img_features.shape : {clip_img_features.shape}')
        return clip_img_features, clip_img_tokens
    
    def get_losses(self, pseudo_words, sampling_results, clip_model, images,
                   *args, **kwargs):
        image_ids = [res.img_id for res in sampling_results]
        device = pseudo_words.device
        # Note: perms = seq
        normed_boxes, spanned_boxes, origin_split, group_split, preds_split_by_perms,\
            seqs_split_split_by_origin, seqs_split_by_group = \
            multi_apply(process_sampling_result_per_image, sampling_results, device=device)
        
        
        # print("start=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        # print(sampling_results)
        # print("box_ids: ")
        # for res in sampling_results:
        #     print(res['box_ids'])
        # print("normed_boxes : ", normed_boxes)
        # # print("spanned_boxes : ", spanned_boxes)
        # print("origin_split : ", origin_split)
        # print("group_split : ", group_split)
        # print("preds_split_by_perms : ", preds_split_by_perms)
        # print("seqs_split_split_by_origin : ", seqs_split_split_by_origin)
        # print("seqs_split_by_group : ", seqs_split_by_group)
        # # origin_split :  [[12, 16, 11]]
        # # group_split :  [[4, 4, 4, 4, 8, 4, 1, 6, 4]]
        # # preds_split_by_perms :  [[2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 1, 3, 3, 2, 2]]
        # # seqs_split_split_by_origin :  [[6, 6, 5]]
        # # seqs_split_by_group :  [[2, 2, 2, 2, 2, 2, 1, 2, 2]]
        # print("=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        
        positions = bbox_xyxy_to_cxcywh(torch.cat(normed_boxes, dim=0))
        position_embeddings = self.positional_embed(positions)
        pseudo_words = pseudo_words + position_embeddings
        word_masks = self._drop_word(pseudo_words)
        
        start_id = 0
        seq_ids = []
        for res in sampling_results:
            seq_ids_ = res['seq_ids']
            for seq_id in seq_ids_:
                seq_ids.append(seq_id + start_id)
            start_id += (max(seq_ids_) + 1)
        seq_ids = torch.tensor(seq_ids, dtype=torch.float32).to(device)   # avoid overflow
        # print("seq_ids: ", seq_ids)
        # # seq_ids = torch.tensor([  4., 740., 740.,  41.,  41., 781., 781., 782., 782., 784., 784.], device='cuda:0')
        # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        
        normed_boxes_split_by_perms = [normed_boxes_.split(preds_split_by_perms_, dim=0)
                                       for normed_boxes_, preds_split_by_perms_
                                       in zip(normed_boxes, preds_split_by_perms)]
        # print("normed_boxes_split_by_perms: ", normed_boxes_split_by_perms)
        # normed_boxes_split_by_perms = [
        #     (tensor([[0., 0., 1., 1.]], device='cuda:0'),
        #      tensor([[0.0000, 0.0000, 0.5263, 0.3571],
        #              [0.4737, 0.3214, 1.0000, 0.6786],
        #              [0.4737, 0.6429, 1.0000, 1.0000]], device='cuda:0'),
        #      tensor([[0.4737, 0.6429, 1.0000, 1.0000],
        #              [0.0000, 0.0000, 0.5263, 0.3571],
        #              [0.4737, 0.3214, 1.0000, 0.6786]], device='cuda:0'),
        #      tensor([[0.0000, 0.4737, 1.0000, 1.0000],
        #              [0.0000, 0.0000, 1.0000, 0.5263]], device='cuda:0'),
        #      tensor([[0.0000, 0.0000, 1.0000, 0.5263],
        #              [0.0000, 0.4737, 1.0000, 1.0000]], device='cuda:0'),
        #      tensor([[0.2904, 0.2097, 1.0000, 1.0000],
        #              [0.0000, 0.0000, 0.3614, 0.2887]], device='cuda:0'),
        #      tensor([[0.0000, 0.0000, 0.3614, 0.2887],
        #              [0.2904, 0.2097, 1.0000, 1.0000]], device='cuda:0'),
        #      tensor([[0.0000, 0.2097, 1.0000, 1.0000],
        #              [0.0000, 0.0000, 1.0000, 0.2887]], device='cuda:0'),
        #      tensor([[0.0000, 0.0000, 1.0000, 0.2887],
        #              [0.0000, 0.2097, 1.0000, 1.0000]], device='cuda:0'),
        #      tensor([[0.0000, 0.0000, 0.3614, 1.0000],
        #              [0.2904, 0.0000, 1.0000, 1.0000]], device='cuda:0'),
        #      tensor([[0.2904, 0.0000, 1.0000, 1.0000],
        #              [0.0000, 0.0000, 0.3614, 1.0000]], device='cuda:0'))]
        
        
        # torch.cat(normed_boxes).split(preds_split_by_perms, dim=0)
        preds_split_by_perms = [p for b in preds_split_by_perms for p in b]
        word_sequences = pseudo_words.split(preds_split_by_perms, dim=0)
        word_masks = word_masks.split(preds_split_by_perms, dim=0)
        word_sequences = [seq.flatten(0, 1)[wm.flatten(0, 1)] for seq, wm in zip(word_sequences, word_masks)]
        context_length = max([seq.shape[0] for seq in word_sequences])
        
        # print("preds_split_by_perms: ", preds_split_by_perms)
        # print("len of word_sequences: ", len(word_sequences))
        # for i in range(len(word_sequences)):
        #     print(f"word_sequences[{i}].shape : {word_sequences[i].shape}")
        # # print("word_masks: ", word_masks)
        # print("context_length: ", context_length)
        
        
        ####################################################################################################
        # Clip Model에서 teacher, student 모델의 text, image feature를 추출
        # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        with autocast():
            text_encoder = clip_model.text_encoder
            # TODO: get local image tokens
            pseudo_text, end_token_ids = text_encoder.prepare_pseudo_text(
                word_sequences,
                context_length=context_length + 2)  # add start and stop token
            clip_text_features, clip_word_tokens = \
                text_encoder.encode_pseudo_text(pseudo_text, end_token_ids,
                                                text_pe=True, normalize=True,
                                                return_word_tokens=True)
            clip_text_features = clip_text_features.float()
            clip_image_features, clip_image_tokens = self._bbox_clip_image(spanned_boxes, images,
                                                                           seqs_split_by_group,
                                                                           normed_boxes_split_by_perms,
                                                                           clip_model)
            
        #     print(f'clip_image_features.shape: {clip_image_features.shape}')
        #     print(f'clip_image_tokens.shape: {clip_image_tokens.shape}')
        #     print(f'clip_text_features.shape: {clip_text_features.shape}')
        #     print(F'length of clip_word_tokens: {len(clip_word_tokens)}')
        # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        ####################################################################################################
       
        ####################################################################################################
        
        global_clip_image_features = self.queues.get_queue('clip_image_features')
        global_clip_text_features = self.queues.get_queue('clip_text_features')
        
        num_queries = clip_text_features.shape[0]
        assert clip_image_features.shape[0] == num_queries
        
        label_mask = seq_ids[None] == seq_ids[:, None]
        label_mask.fill_diagonal_(False)
        # print(f'label_mask: {label_mask}')
        # label_mask = torch.tensor([[False, False, False, False, False],
        #                            [False, False,  True, False, False],
        #                            [False,  True, False, False, False],
        #                            [False, False, False, False,  True],
        #                            [False, False, False,  True, False]])
        # mask same synced_img

        
        
        
        img_ids = [torch.tensor(sum(b) * [img_id])
                   for b, img_id in zip(seqs_split_split_by_origin,
                                        image_ids)]
        img_ids = torch.cat(img_ids).to(device)
        global_text_feature_img_ids = global_clip_text_features[..., -1]
        global_image_feature_img_ids = global_clip_image_features[..., -1]

        # text features as queries
        image_keys = torch.cat([clip_image_features, global_clip_image_features[..., :-1]], dim=0)
        # print("image_keys.shape : ", image_keys.shape)
        similarity_matrix_0 = self.bag_temp * clip_text_features @ image_keys.T
        similarity_matrix_0[:, :num_queries][label_mask] = float('-inf')
        if global_image_feature_img_ids.shape[0] > 0:
            img_id_mask_0 = img_ids[:, None] == global_image_feature_img_ids[None]
            assert similarity_matrix_0[:, num_queries:].shape == img_id_mask_0.shape, \
                f"image_ids: {img_ids}, {image_ids}, {len(seqs_split_split_by_origin)}"
            similarity_matrix_0[:, num_queries:][img_id_mask_0] = float('-inf')
            
        # image features as queries
        text_keys = torch.cat([clip_text_features, global_clip_text_features[..., :-1]], dim=0)
        # print("text_keys.shape : ", text_keys.shape)
        similarity_matrix_1 = self.bag_temp * clip_image_features @ text_keys.T
        similarity_matrix_1[:, :num_queries][label_mask] = float('-inf')
        if global_text_feature_img_ids.shape[0] > 0:
            img_id_mask_1 = img_ids[:, None] == global_text_feature_img_ids[None]
            similarity_matrix_1[:, num_queries:][img_id_mask_1] = float('-inf')
            
        ####################################################################################################
        # print("=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        # print(f"Similarity Matrix 0 (Text as Query):")
        # print("similarity_matrix_0.shape : ", similarity_matrix_0.shape)
        # print(similarity_matrix_0)
    
        # print(f"Similarity Matrix 1 (Image as Query):")
        # print("similarity_matrix_1.shape : ", similarity_matrix_1.shape)
        # print(similarity_matrix_1)
        
        
        # print(clip_image_features.shape, global_clip_image_features.shape)
        # print(image_keys.shape)
        
        
        # import pdb; pdb.set_trace()
        
        # sim_0 = similarity_matrix_0/self.bag_temp
        # sim_1 = similarity_matrix_1/self.bag_temp
        # print(f'Similarity Matrix 0: \n{sim_0}')
        # print(f'Similarity Matrix 1: \n{sim_1}')
        

        # # 시각화 (옵션)
        # # self.plot_similarity_matrix(similarity_matrix_0, "Similarity Matrix 0 (Text as Query)")
        # # self.plot_similarity_matrix(similarity_matrix_1, "Similarity Matrix 1 (Image as Query)")
        
        # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        # print(ddd)
        ####################################################################################################
        
        with torch.autograd.set_detect_anomaly(True):
            # Label Smoothing
            if self.label_smoothing[0]:
                #####################################################################
                # (for comparison)
                # label = torch.arange(num_queries).to(device)
                # 
                # loss = 0.5 * F.cross_entropy(similarity_matrix_0, label) \
                #     + 0.5 * F.cross_entropy(similarity_matrix_1, label)
                # print(loss)
                #####################################################################
                
                # First, find the same origin sequences and mask them
                mutual_origin = []
                start_id = 0
                for b in seqs_split_split_by_origin:
                    for seqs in b:
                        mutual_origin.extend([seqs+start_id for _ in range(seqs)])
                        start_id += seqs
                mutual_origin = torch.tensor(mutual_origin)
                soft_negative_label = mutual_origin[None] == mutual_origin[:, None]
                # print(f'soft_negative_label : {soft_negative_label}')
                
                if self.label_smoothing_positional_similarity[0]:
                    label_matrix = torch.zeros_like(similarity_matrix_0)
                    
                    box_ids_list= []
                    for res in sampling_results:
                        for ori in res['box_ids']:
                            for perm in ori:
                                box_ids_list.append(perm)
                    
                    min_smoothing = self.label_smoothing_alpha[0][0]
                    max_smoothing = self.label_smoothing_alpha[0][1]
                    
                    for i in range(label_matrix.shape[0]):
                        for j in range(label_matrix.shape[0]):
                            if soft_negative_label[i, j]:
                                bag1 = set(box_ids_list[i]) - {4}
                                bag2 = set(box_ids_list[j]) - {4}
                                if len(bag1)==0 or len(bag2)==0:
                                    similarity = 1
                                else:
                                    similarity = len(bag1.intersection(bag2)) / len(bag1.union(bag2))
                                label_matrix[i, j] = min_smoothing + (max_smoothing - min_smoothing) * similarity
                    
                    label_matrix[:, :num_queries][label_mask] = 0               # permutated sequences
                    label_matrix.fill_diagonal_(1)                              # itself
                else :
                    label_matrix = torch.zeros_like(similarity_matrix_0)
                    label_matrix[:, :num_queries][soft_negative_label] = self.label_smoothing_alpha[0]    # sequences from the same origin
                    label_matrix[:, :num_queries][label_mask] = 0               # permutated sequences
                    label_matrix.fill_diagonal_(1)                              # itself
                # print(label_matrix)

                logprobs_0 = F.log_softmax(similarity_matrix_0, dim=-1)
                logprobs_1 = F.log_softmax(similarity_matrix_1, dim=-1)
                
                extended_label_mask = F.pad(torch.tensor(label_mask), (0, similarity_matrix_0.shape[1]-similarity_matrix_0.shape[0], 0, 0), value=False)
                logprobs_0 = logprobs_0 * ~extended_label_mask
                logprobs_1 = logprobs_1 * ~extended_label_mask
                
                # logprobs_0[:, :num_queries][label_mask] = 0                 # -inf(permutation)계산시 0으로 대체 -> 안그럼 nan
                # logprobs_1[:, :num_queries][label_mask] = 0
                
                res_0 = label_matrix * -logprobs_0
                res_1 = label_matrix * -logprobs_1

                res_0 = torch.nan_to_num(res_0, nan=0.0)
                res_1 = torch.nan_to_num(res_1, nan=0.0)
                
                loss = 0.5 * res_0.sum(dim=-1).mean() \
                    + 0.5 * res_1.sum(dim=-1).mean()
                    
                # print(loss)
            else:
                label = torch.arange(num_queries).to(device)
                # print('label:', label)
                
                loss = 0.5 * F.cross_entropy(similarity_matrix_0, label) \
                    + 0.5 * F.cross_entropy(similarity_matrix_1, label)
                
        losses = dict(loss_bag=loss * self.bag_weight)
        
        # Enqueue
        queues_update = dict(clip_text_features=torch.cat([clip_text_features,
                                                      img_ids.view(-1, 1)], dim=-1).detach(),
                             clip_image_features=torch.cat([clip_image_features,
                                                      img_ids.view(-1, 1)], dim=-1).detach())
        
        ####################################################################################################
        visualize = False
        if self.similarity_preserving_weight > 0.0:
            
            # 11/12
            # 의미적인 visualize 결과 애매하면 distribution 이동하는거 보여주기
            # 우리근데 논문에 앞으로 다 45k로 내는건가 수치
            # inference code 수정해서 이것저것 할 수 있도록 하는 방법
            ##################################################################
            # similarity-preserving
            #  - matrix logartihm 구현 + similarity preserving 할 때 1/b^2이랑 논문의 lambda값 확인하기 
            # label smoothing
            #  - positional similarity 수치적으로 말고 상대적으로 멀어지게 만드는 방법
            # future works
            #  - multi-bag
            
            if visualize:
                torch.set_printoptions(profile="full")
                
                now = datetime.now()
                os.mkdir(f'/work/ovdet/work_dirs/jh/aff_matrices_sp45k/{now}')
                img_name = os.path.basename(sampling_results[0].img_path)
                img_name = f'{now}/{img_name}'
                
                # [1] label_mask 처리
                cumulated_label_mask = self.queues.get_queue('label_mask')
                if cumulated_label_mask.shape[0]==0 or cumulated_label_mask.shape[0]==1:
                    new_cumulated_label_mask = label_mask
                else:
                    new_cumulated_label_mask = torch.ones(label_mask.shape[0]+cumulated_label_mask.shape[0], label_mask.shape[1]+cumulated_label_mask.shape[1], dtype=torch.bool)
                    new_cumulated_label_mask[:label_mask.shape[0], :label_mask.shape[1]] = label_mask
                    new_cumulated_label_mask[label_mask.shape[0]:, label_mask.shape[1]:] = cumulated_label_mask
                queues_update.update(label_mask=new_cumulated_label_mask.detach())
                
                
                # [2] 같은 image, 같은 region에 대한 box_coords
                print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
                vis_box_coords = True
                if vis_box_coords:
                    # seqs_split_split_by_origin = [[6, 6, 5]]
                    box_coords = [] # visualize용으로 같은 region proposal에서 나온 bag들끼리 묶어주기 : [(10, 10, 10, 10), (20, 20, 5, 5)]
                    image_box_coords = [] # 이미지끼리 묶어주기 
                    start = 0
                    start2 = 0
                    print(f'seqs_split_split_by_origin: {seqs_split_split_by_origin}')
                    for b in seqs_split_split_by_origin:
                        box_coord = []
                        for seqs in b:
                            box_coord.append((start, start, seqs, seqs))
                            start += seqs
                        box_coords.extend(box_coord)
                        image_box_coords.append((start2, start2, start, start))
                        start2 += start
                    print(f'box_coords: {box_coords}')
                    print(f'image_box_coords: {image_box_coords}')

                    cumulated_box_coords = self.queues.get_queue('box_coords')
                    cumulated_box_coords = cumulated_box_coords.tolist()
                    cumulated_box_coords = [tuple(box_coord) for box_coord in cumulated_box_coords]
                    if len(cumulated_box_coords)==0 or len(cumulated_box_coords)==1:
                        cumulated_box_coords = box_coords
                    else:
                        print(cumulated_box_coords)
                        start = box_coords[-1][0] + box_coords[-1][2]
                        cumulated_box_coords = [(x+start, y+start, w, h) for (x, y, w, h) in cumulated_box_coords]
                        cumulated_box_coords = box_coords + cumulated_box_coords
                    queues_update.update(box_coords=torch.tensor(cumulated_box_coords).detach())
                    print(torch.tensor(cumulated_box_coords))
                    
                    cumulated_image_box_coords = self.queues.get_queue('image_box_coords')
                    cumulated_image_box_coords = cumulated_image_box_coords.tolist()
                    cumulated_image_box_coords = [tuple(image_box_coord) for image_box_coord in cumulated_image_box_coords]
                    if len(cumulated_image_box_coords[0]) != 4:
                        cumulated_image_box_coords = image_box_coords
                    else:
                        print(cumulated_image_box_coords)
                        start = image_box_coords[-1][0] + image_box_coords[-1][2]
                        cumulated_image_box_coords = [(x+start, y+start, w, h) for (x, y, w, h) in cumulated_image_box_coords]
                        cumulated_image_box_coords = image_box_coords + cumulated_image_box_coords
                    queues_update.update(image_box_coords=torch.tensor(cumulated_image_box_coords).detach())
                    print(torch.tensor(cumulated_image_box_coords))
                else:
                    box_coords = False
                    cumulated_box_coords = False
                    image_box_coords = False
                    cumulated_image_box_coords = False
                print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")

                

                # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
                ####################################################################################################
                student_similarity_matrix = clip_text_features @ clip_text_features.T
                teacher_similarity_matrix = clip_image_features @ clip_image_features.T
            
                visualize_two_matrices(student_similarity_matrix, teacher_similarity_matrix, img_name+'_ori', box_coords=box_coords, image_box_coords=image_box_coords)
            
                student_similarity_matrix = self.bag_temp * clip_text_features @ clip_text_features.T
                student_similarity_matrix.fill_diagonal_(float('-inf'))
                student_similarity_matrix[label_mask] = float('-inf')
                student_similarity_matrix = 0.5 * F.softmax(student_similarity_matrix, dim=0) \
                    + 0.5 * F.softmax(student_similarity_matrix, dim=1)
                    
                teacher_similarity_matrix = self.bag_temp * clip_image_features @ clip_image_features.T
                teacher_similarity_matrix.fill_diagonal_(float('-inf'))
                teacher_similarity_matrix[label_mask] = float('-inf')
                teacher_similarity_matrix = 0.5 * F.softmax(teacher_similarity_matrix, dim=0) \
                    + 0.5 * F.softmax(teacher_similarity_matrix, dim=1)
            
                # print(f'student similarity (w/o queue) : {student_similarity_matrix}')
                # print(f'teacher similarity (w/o queue) : {teacher_similarity_matrix}')
                visualize_two_matrices(student_similarity_matrix, teacher_similarity_matrix, img_name+'_ori+softmax', box_coords=box_coords, image_box_coords=image_box_coords)
                
                ####################################################################################################
                # # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
                # print("queue included")
                student_similarity_matrix = text_keys @ text_keys.T
                teacher_similarity_matrix = image_keys @ image_keys.T
                
                # print(teacher_similarity_matrix)
                visualize_two_matrices(student_similarity_matrix, teacher_similarity_matrix, img_name+'_queue', box_coords=cumulated_box_coords, image_box_coords=cumulated_image_box_coords)
                
                # # print(f'student similarity (w/ queue) : {student_similarity_matrix}')
                # # print(f'teacher similarity (w/ queue) : {teacher_similarity_matrix}')
                
                student_similarity_matrix = self.bag_temp * text_keys @ text_keys.T
                student_similarity_matrix.fill_diagonal_(float('-inf'))
                student_similarity_matrix[new_cumulated_label_mask] = float('-inf')
                student_similarity_matrix = 0.5 * F.softmax(student_similarity_matrix, dim=0) \
                    + 0.5 * F.softmax(student_similarity_matrix, dim=1)
                
                teacher_similarity_matrix = self.bag_temp * image_keys @ image_keys.T
                teacher_similarity_matrix.fill_diagonal_(float('-inf'))
                teacher_similarity_matrix[new_cumulated_label_mask] = float('-inf')
                teacher_similarity_matrix = 0.5 * F.softmax(teacher_similarity_matrix, dim=0) \
                    + 0.5 * F.softmax(teacher_similarity_matrix, dim=1)
                # print(teacher_similarity_matrix)
                
                visualize_two_matrices(student_similarity_matrix, teacher_similarity_matrix, img_name+'_queue+softmax', box_coords=cumulated_box_coords, image_box_coords=cumulated_image_box_coords)
                ####################################################################################################
                
                
                # print(ddd)
                
                # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
                # # print(f'image keys : {image_keys}')
                # # print(f'text keys : {text_keys}')
            
            
            # student_similarity_matrix = clip_text_features @ clip_text_features.T
            # teacher_similarity_matrix = clip_image_features @ clip_image_features.T
            
            student_similarity_matrix = text_keys @ text_keys.T
            teacher_similarity_matrix = image_keys @ image_keys.T
            
            student_similarity_matrix = F.normalize(student_similarity_matrix, p=2, dim=-1)
            teacher_similarity_matrix = F.normalize(teacher_similarity_matrix, p=2, dim=-1)

            # log_mat1 = matrix_logarithm(student_similarity_matrix)
            # log_mat2 = matrix_logarithm(teacher_similarity_matrix)
            # print("Log-Euclidean Similarity Distillation Loss")
            # print(log_mat1.shape, log_mat2.shape)
            
            distance_type = 'L2'
            if distance_type == 'Log-Euclidean':
                dist = matrix_logarithm(student_similarity_matrix) - matrix_logarithm(teacher_similarity_matrix)
                # dist = torch.clamp(dist, min=-10.0, max=10.0))
                
                # print("Student Log Map:", matrix_logarithm(student_similarity_matrix))
                # print("Teacher Log Map:", matrix_logarithm(teacher_similarity_matrix))
                # print("Difference (Student - Teacher):", dist)
                dist = torch.norm(dist, p='fro')
            elif distance_type == 'L2':
                dist = student_similarity_matrix - teacher_similarity_matrix
                dist = torch.norm(dist, p='fro')
                # print("Frobenius Norm of Difference:", dist)
            elif distance_type == 'L1':
                dist = student_similarity_matrix - teacher_similarity_matrix
                dist = torch.norm(dist, p='1')
                

            loss = dist ** 2
            loss /= (teacher_similarity_matrix.shape[0] ** 2)
            # loss /= dist.shape[0]
            
            # loss = torch.clamp(loss, max=100.0)
            # print("Log-Euclidean Similarity Distillation Loss:", loss)
            
            losses.update(loss_similarity_preserving=loss * self.similarity_preserving_weight)

            # print(loss)
        
        if visualize : 
            if similarity_matrix_0.shape[1] > 300:
                print(ddd)
        
        
        ####################################################################################################
        if self.multi_proposal_bag_weight > 0.0:
            pass
        
        ####################################################################################################
            
        if self.single_weight > 0.0:
            preds_split_by_batch = [n.shape[0] for n in normed_boxes]
            img_ids = [torch.tensor(b * [img_id])
                       for b, img_id in zip(preds_split_by_batch,
                                            image_ids)]
            img_ids = torch.cat(img_ids).to(device)
            normed_boxes = torch.cat(normed_boxes, dim=0).split(preds_split_by_perms, dim=0)
            clip_patch_features = F.normalize(roi_align(
                clip_image_tokens, bbox2roi(normed_boxes).to(clip_image_tokens.dtype), (1, 1),
                float(clip_image_tokens.shape[-1]), 2, 'avg', True)[..., 0, 0], dim=-1)
            num_words_per_pred = [wm.sum(-1).tolist() for wm in word_masks]
            clip_word_features = [tk.split(spl) for (tk, spl)
                                  in zip(clip_word_tokens, num_words_per_pred)]
            clip_word_features = F.normalize(torch.stack([feat.mean(0).float()
                                                          for feats in clip_word_features
                                                          for feat in feats], dim=0), dim=-1)
            start_id = 0
            box_ids = []
            for res in sampling_results:
                for ori in res['box_ids']:
                    
                    box_ids_per_ori = [torch.tensor(perm, dtype=torch.float32)
                                       for perm in ori]   # avoid overflow
                    try:
                        box_ids_per_ori = torch.cat(box_ids_per_ori) + start_id
                    except RuntimeError:
                        from mmengine.logging import print_log
                        print_log(f'{box_ids_per_ori}, {start_id}')
                        exit()
                    start_id += (box_ids_per_ori.max().item() + 1)
                    box_ids.append(box_ids_per_ori)
            box_ids = torch.cat(box_ids).to(device)
            
            global_clip_word_features = self.queues.get_queue('clip_word_features')
            global_clip_patch_features = self.queues.get_queue('clip_patch_features')

            global_word_feature_img_ids = global_clip_word_features[..., -1]
            global_patch_feature_img_ids = global_clip_patch_features[..., -1]

            num_queries = clip_patch_features.shape[0]
            assert num_queries == clip_word_features.shape[0]

            # text features as queries
            image_keys = torch.cat([clip_patch_features, global_clip_patch_features[..., :-1]])
            similarity_matrix_0 = self.single_temp * clip_word_features @ image_keys.T
            if global_patch_feature_img_ids.shape[0] > 0:
                img_id_mask_0 = img_ids[:, None] == global_patch_feature_img_ids[None]
                similarity_matrix_0[:, num_queries:][img_id_mask_0] = float('-inf')
                
            # image features as queries
            text_keys = torch.cat([clip_word_features, global_clip_word_features[..., :-1]])
            similarity_matrix_1 = self.single_temp * clip_patch_features @ text_keys.T
            if global_word_feature_img_ids.shape[0] > 0:
                img_id_mask_1 = img_ids[:, None] == global_word_feature_img_ids[None]
                similarity_matrix_1[:, num_queries:][img_id_mask_1] = float('-inf')
            
            labels = torch.arange(num_queries, device=device)
            label_mask = box_ids[None] == box_ids[:, None]
            label_mask.fill_diagonal_(False)

            similarity_matrix_0[:, :num_queries][label_mask] = float('-inf')
            similarity_matrix_1[:, :num_queries][label_mask] = float('-inf')
            
            loss = F.cross_entropy(similarity_matrix_0, labels) * 0.5 \
                   + F.cross_entropy(similarity_matrix_1, labels) * 0.5
            losses.update(loss_single=loss * self.single_weight)

            queues_update.update(clip_word_features=torch.cat([clip_word_features,
                                                               img_ids.view(-1, 1)], dim=-1).detach(),
                                 clip_patch_features=torch.cat([clip_patch_features,
                                                                img_ids.view(-1, 1)], dim=-1).detach())
            self.queues.dequeue_and_enqueue(queues_update)

        return losses
