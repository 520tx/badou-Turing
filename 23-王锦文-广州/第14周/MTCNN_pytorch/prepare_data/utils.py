import numpy as np
import torch




def py_nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        predicted boxes
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

#将矩形的bounding box 转化为正方形 正方形的变长为矩形的最长边
def convert_to_square(bbox):
    """Convert bbox to square

    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox

    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox


def torch_nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4]. 
                tensor([[0.7624, 0.1910, 0.8234, 0.3271],
                        [0.7760, 0.1734, 0.8310, 0.3357],
                        [0.3976, 0.4168, 0.5528, 0.5226]])  
        scores: (tensor) The class predscores for the img, Shape:[num_priors].每个location对应的阈值
                其中num_priors是满足阈值要求的个数这里为3  score:tensor([0.0117, 0.0131, 0.0114]) [num,]
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    #print("keep :{}".format(keep))
    if boxes.numel() == 0:#torch.numel() 返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1 + 1, y2 - y1 + 1)
    _, idx = scores.sort(0)  # 升序排列score   #tensor([0.0114, 0.0117, 0.0131]) idx:tensor([2, 0, 1])
    #print("scores sort v:{} idx:{}".format(v,idx))

    # idx = idx[-top_k:]  # 需要取出topk个框 idx:tensor([2, 0, 1]),不足topk的取原来的长度

    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    count = 0

    while idx.numel() > 0:           # torch.numel()返回张量元素个数
        i = idx[-1]  # index of current largest val   获取当前最大sorces数值的索引 1
        keep[count] = i
        #print("keep 222 {}".format(keep))
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view 移除最后一个最大值的索引
        # load bboxes of next highest vals
        #取出除阈值最高的box之外，所有的box的x1,y1,x2,y2赋值给xx1，yy1，xx2，yy2
        torch.index_select(x1, 0, idx, out=xx1)         #xx1:tensor([0.3976, 0.7624])
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        #求剩下的box与阈值最高的box相比，左上角的较大值 与右下角的较小值
        xx1 = torch.clamp(xx1, min=x1[i])               #xx1:tensor([0.7760, 0.7760])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        #右下角坐标-左上角坐标
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        #求得剩下的box与阈值最高的box 交集
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # rem_areas 取出剩余的box 的面积 
        union = (rem_areas - inter) + area[i]#area[i] 当前阈值最大的box的面积 计算并集
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        #print(IoU.le(overlap))   tensor([1, 0]
        idx = idx[IoU.le(overlap)]#从剩下的box中 取出iou的值小于阈值的box对应的索引
        #print(idx.numel())1
    #print(keep,count)  #tensor([1, 2, 0]) 2
    return keep, count