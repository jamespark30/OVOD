import torch
import torch.nn as nn
from .builder import QUEUE


@QUEUE.register_module()
class Queues(nn.Module):
    def __init__(self, names, lengths, emb_dim=512, id_length=4):
        super(Queues, self).__init__()
        self.names = names
        self.lengths = lengths
        self.emb_dim = emb_dim
        self.id_length = id_length
        self._init_queues()

    def _init_queues(self):
        attr_names = self.names
        queue_lengths = self.lengths
        for n in attr_names:
            self.register_buffer(n, -torch.ones(1, self.emb_dim + self.id_length),
                                 persistent=False)
        self.queue_lengths = {n: queue_lengths[i] for i, n in enumerate(attr_names)}

    @torch.no_grad()
    def dequeue_and_enqueue(self, queue_update):
        for k, feat in queue_update.items():
            if k == 'label_mask':
                queue_length = self.queue_lengths[k]
                queue_value = getattr(self, k)
                queue_value.data = feat[:queue_length][:queue_length]
            elif k == 'box_coords' or k == 'image_box_coords':
                queue_length = self.queue_lengths[k]
                sliced_output = []
                for box in feat:
                    if box[0] + box[2] > queue_length:
                        break
                    sliced_output.append(box.tolist())
                queue_value = getattr(self, k)
                queue_value.data = torch.tensor(sliced_output)
            else:
                queue_length = self.queue_lengths[k]
                valid = (feat[:, self.emb_dim:] >= 0).sum(-1) > 0   # valid label
                if valid.sum() == 0:
                    continue
                feat = feat[valid]
                feat = feat[:queue_length]
                in_length = feat.shape[0]
                queue_value = getattr(self, k)
                current_length = queue_value.shape[0]
                kept_length = min(queue_length - in_length, current_length)

                queue_value.data = torch.cat([feat, queue_value[:kept_length]])

    @torch.no_grad()
    def get_queue(self, key):
        # print(key)
        if key == 'label_mask' or key == 'box_coords' or key == 'image_box_coords':
            value = getattr(self, key)
            return value
        else:
            value = getattr(self, key)
            valid = (value[:, self.emb_dim:] >= 0).sum(-1) > 0  # valid label
            return value[valid]
