
import torch


def get_inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7):
    if not xywh:
        def xyxy2xywh(x):
            y = torch.empty_like(x)
            y[..., 0] = (x[..., 0] + x[..., 2]) / 2
            y[..., 1] = (x[..., 1] + x[..., 3]) / 2
            y[..., 2] = x[..., 2] - x[..., 0]
            y[..., 3] = x[..., 3] - x[..., 1]
            return y

        box1, box2 = xyxy2xywh(box1), xyxy2xywh(box2)

    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)

    b1_x1 = x1 - (w1 * ratio) / 2
    b1_x2 = x1 + (w1 * ratio) / 2
    b1_y1 = y1 - (h1 * ratio) / 2
    b1_y2 = y1 + (h1 * ratio) / 2

    b2_x1 = x2 - (w2 * ratio) / 2
    b2_x2 = x2 + (w2 * ratio) / 2
    b2_y1 = y2 - (h2 * ratio) / 2
    b2_y2 = y2 + (h2 * ratio) / 2

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps

    return inter / union


def bbox_inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7, scale=0.1):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)

        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2

        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inner_iou = get_inner_iou(box1, box2, xywh=xywh, eps=eps, ratio=ratio)

    ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
    center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    center_distance = hh * center_distance_x + ww * center_distance_y
    distance = center_distance / c2

    omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)

    shape_cost = torch.pow(1 - torch.exp(-omiga_w), 4) + \
                 torch.pow(1 - torch.exp(-omiga_h), 4)

    return inner_iou - distance - 0.5 * shape_cost
