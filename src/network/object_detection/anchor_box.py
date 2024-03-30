import torch
from torch import nn
from d2l import torch as d2l
# 锚框
# 锚框是一组预定义的边界框，用于检测图像中的对象。
# 为了简单起见，我们假设图像中的所有对象形状都是矩形。
# 在目标检测时，我们将每个锚框视为一个单独的图像样本。
# 为了标记任何锚框，我们需要了解该锚框与真实边界框之间的重叠度。

# IoU-交并比
# 交并比（IoU）是两个边界框的交集与并集之间的比率。
# 交集是两个边界框的交集，而并集是两个边界框的并集。
# 交并比的取值范围在0和1之间：0表示两个边界框没有重叠，1表示两个边界框完全重叠。

# 赋予锚框标号
# 在目标检测中，我们通常将每个锚框视为一个样本。
# 为了训练目标检测模型，我们需要为每个锚框标注两类标签：一类是锚框所含目标的类别，另一类是真实边界框相对于锚框的偏移量。
# 我们可能会生成大量的锚框，以覆盖图像上的所有区域，这可能导致大量的负类锚框。
torch.set_printoptions(2)


def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框。"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height
    steps_w = 1.0 / in_width

    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    anchor_manipulations = torch.stack(
        (-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                           dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def show_bbox(axes, bboxes, labels=None, colors=None):
    """显示所有边界框。"""
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center',
                      ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0),
                      backgroundcolor='none')


def box_iou(boxes1, boxes2):
    """计算两个锚框或边界框列表中成对的交并比。"""
    def box_area(boxes): return ((boxes[:, 2] - boxes[:, 0]) *
                                 (boxes[:, 3] - boxes[:, 1]))
    # `boxes1`, `boxes2`, `areas1`, `areas2`的形状:
    # `boxes1`：(boxes1的数量, 4),
    # `boxes2`：(boxes2的数量, 4),
    # `areas1`：(boxes1的数量,),
    # `areas2`：(boxes2的数量,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # `inter_upperlefts`, `inter_lowerrights`, `inters`的形状:
    # (boxes1的数量, boxes2的数量, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # `inter_areas` and `union_areas`的形状: (boxes1的数量, boxes2的数量)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框。"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素 x_ij 是锚框 i 和真实边界框 j 的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full(
        (num_anchors,), -1, dtype=torch.long, device=device)
    # 根据阈值，决定是否分配真实边界框
    max_ious, indices = jaccard.max(1)
    anc_i = torch.nonzero(max_ious >= 0.5).reshape(-1)
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换。"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10.0 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5.0 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset


def multibox_target(anchors, labels):
    """使用真实边界框标记锚框。"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = (
            (anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        class_labels = torch.zeros(
            num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros(
            (num_anchors, 4), dtype=torch.float32, device=device)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框。"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10.0) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5.0) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
# nms是非极大值抑制的缩写。它是目标检测中一种常用的后处理技术。


def nms(boxes, scores, iou_threshold):
    """对预测边界框的置信度进行排序。"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1:
            break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)


def multiscale_object_detection(data, sizes, ratios, num_classes, iou_threshold):
    """使用多尺度目标检测测试图像。"""
    device, net = next(iter(data)), d2l.ssd_model(num_classes)
    net = net.to(device)
    net.eval()
    scales, frames = [1, 0.5, 0.25], []
    for scale in scales:
        frame = d2l.resize(data, scale)
        frames.append(frame)
    anchors = multibox_prior(frames, sizes, ratios)
    batch_size = 32
    data = torch.stack(frames, 0).to(device)
    anchors = anchors.to(device)
    cls_preds, bbox_preds = net(data)
    cls_probs = cls_preds.softmax(-1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output) if row.sum() != 0]
    output = output[idx]
    fig = d2l.plt.imshow(data[0])
    for i in idx:
        show_bbox(fig.axes, output[i, :, :4] * scales[i],
                  d2l.classes[int(output[i, 0, 4])], 'w')


if __name__ == '__main__':
    img = d2l.plt.imread('../img/catdog.jpg')
    h, w = img.shape[0:2]
    print(h, w)
    X = torch.rand(size=(1, 3, h, w))
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print(Y.shape)
    # output:
    # torch.Size([1, 2042040, 4])
