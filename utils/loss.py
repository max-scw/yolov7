# Loss functions

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.functional import one_hot, binary_cross_entropy_with_logits

from utils.general import bbox_iou, bbox_alpha_iou, box_iou, box_giou, box_diou, box_ciou, xywh2xyxy
from utils.torch_utils import is_parallel

from typing import List, Union


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class SigmoidBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, bin_count=10, min=0.0, max=1.0, reg_scale=2.0, use_loss_regression=True, use_fw_regression=True,
                 BCE_weight=1.0, smooth_eps=0.0):
        super(SigmoidBin, self).__init__()

        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0

        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight

        start = min + (self.scale / 2.0) / self.bin_count
        end = max - (self.scale / 2.0) / self.bin_count
        step = self.scale / self.bin_count
        self.step = step
        # print(f" start = {start}, end = {end}, step = {step} ")

        bins = torch.range(start, end + 0.0001, step).float()
        self.register_buffer('bins', bins)

        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps

        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (
        pred.shape[-1], self.length)

        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1:(1 + self.bin_count)]

        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]

        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min, max=self.max)

        return result

    def training_loss(self, pred, target):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (
        pred.shape[-1], self.length)
        assert pred.shape[0] == target.shape[0], 'pred.shape=%d is not equal to the target.shape=%d' % (
        pred.shape[0], target.shape[0])
        device = pred.device

        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1:(1 + self.bin_count)]

        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)

        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias

        target_bins = torch.full_like(pred_bin, self.cn, device=device)  # targets
        n = pred.shape[0]
        target_bins[range(n), bin_idx] = self.cp

        loss_bin = self.BCEbins(pred_bin, target_bins)  # BCE

        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)  # MSE        
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin

        out_result = result.clamp(min=self.min, max=self.max)

        return loss, out_result


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class RankSort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta_RS=0.50, eps=1e-10):

        classification_grads = torch.zeros(logits.shape).cuda()

        # Filter fg logits
        fg_labels = (targets > 0.)
        fg_logits = logits[fg_labels]
        fg_targets = targets[fg_labels]
        fg_num = len(fg_logits)

        # Do not use bg with scores less than minimum fg logit
        # since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits) - delta_RS
        relevant_bg_labels = ((targets == 0) & (logits >= threshold_logit))

        relevant_bg_logits = logits[relevant_bg_labels]
        relevant_bg_grad = torch.zeros(len(relevant_bg_logits)).cuda()
        sorting_error = torch.zeros(fg_num).cuda()
        ranking_error = torch.zeros(fg_num).cuda()
        fg_grad = torch.zeros(fg_num).cuda()

        # sort the fg logits
        order = torch.argsort(fg_logits)
        # Loops over each positive following the order
        for ii in order:
            # Difference Transforms (x_ij)
            fg_relations = fg_logits - fg_logits[ii]
            bg_relations = relevant_bg_logits - fg_logits[ii]

            if delta_RS > 0:
                fg_relations = torch.clamp(fg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
                bg_relations = torch.clamp(bg_relations / (2 * delta_RS) + 0.5, min=0, max=1)
            else:
                fg_relations = (fg_relations >= 0).float()
                bg_relations = (bg_relations >= 0).float()

            # Rank of ii among pos and false positive number (bg with larger scores)
            rank_pos = torch.sum(fg_relations)
            FP_num = torch.sum(bg_relations)

            # Rank of ii among all examples
            rank = rank_pos + FP_num

            # Ranking error of example ii. target_ranking_error is always 0. (Eq. 7)
            ranking_error[ii] = FP_num / rank

            # Current sorting error of example ii. (Eq. 7)
            current_sorting_error = torch.sum(fg_relations * (1 - fg_targets)) / rank_pos

            # Find examples in the target sorted order for example ii
            iou_relations = (fg_targets >= fg_targets[ii])
            target_sorted_order = iou_relations * fg_relations

            # The rank of ii among positives in sorted order
            rank_pos_target = torch.sum(target_sorted_order)

            # Compute target sorting error. (Eq. 8)
            # Since target ranking error is 0, this is also total target error
            target_sorting_error = torch.sum(target_sorted_order * (1 - fg_targets)) / rank_pos_target

            # Compute sorting error on example ii
            sorting_error[ii] = current_sorting_error - target_sorting_error

            # Identity Update for Ranking Error
            if FP_num > eps:
                # For ii the update is the ranking error
                fg_grad[ii] -= ranking_error[ii]
                # For negatives, distribute error via ranking pmf (i.e. bg_relations/FP_num)
                relevant_bg_grad += (bg_relations * (ranking_error[ii] / FP_num))

            # Find the positives that are misranked (the cause of the error)
            # These are the ones with smaller IoU but larger logits
            missorted_examples = (~ iou_relations) * fg_relations

            # Denominotor of sorting pmf
            sorting_pmf_denom = torch.sum(missorted_examples)

            # Identity Update for Sorting Error
            if sorting_pmf_denom > eps:
                # For ii the update is the sorting error
                fg_grad[ii] -= sorting_error[ii]
                # For positives, distribute error via sorting pmf (i.e. missorted_examples/sorting_pmf_denom)
                fg_grad += (missorted_examples * (sorting_error[ii] / sorting_pmf_denom))

        # Normalize gradients by number of positives
        classification_grads[fg_labels] = (fg_grad / fg_num)
        classification_grads[relevant_bg_labels] = (relevant_bg_grad / fg_num)

        ctx.save_for_backward(classification_grads)

        return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1, = ctx.saved_tensors
        return g1 * out_grad1, None, None, None


class aLRPLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, regression_losses, delta=1., eps=1e-5):
        classification_grads = torch.zeros(logits.shape).cuda()

        # Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        # Do not use bg with scores less than minimum fg logit
        # since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits) - delta

        # Get valid bg logits
        relevant_bg_labels = ((targets == 0) & (logits >= threshold_logit))
        relevant_bg_logits = logits[relevant_bg_labels]
        relevant_bg_grad = torch.zeros(len(relevant_bg_logits)).cuda()
        rank = torch.zeros(fg_num).cuda()
        prec = torch.zeros(fg_num).cuda()
        fg_grad = torch.zeros(fg_num).cuda()

        max_prec = 0
        # sort the fg logits
        order = torch.argsort(fg_logits)
        # Loops over each positive following the order
        for ii in order:
            # x_ij s as score differences with fgs
            fg_relations = fg_logits - fg_logits[ii]
            # Apply piecewise linear function and determine relations with fgs
            fg_relations = torch.clamp(fg_relations / (2 * delta) + 0.5, min=0, max=1)
            # Discard i=j in the summation in rank_pos
            fg_relations[ii] = 0

            # x_ij s as score differences with bgs
            bg_relations = relevant_bg_logits - fg_logits[ii]
            # Apply piecewise linear function and determine relations with bgs
            bg_relations = torch.clamp(bg_relations / (2 * delta) + 0.5, min=0, max=1)

            # Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos = 1 + torch.sum(fg_relations)
            FP_num = torch.sum(bg_relations)
            # Store the total since it is normalizer also for aLRP Regression error
            rank[ii] = rank_pos + FP_num

            # Compute precision for this example to compute classification loss
            prec[ii] = rank_pos / rank[ii]
            # For stability, set eps to a infinitesmall value (e.g. 1e-6), then compute grads
            if FP_num > eps:
                fg_grad[ii] = -(torch.sum(fg_relations * regression_losses) + FP_num) / rank[ii]
                relevant_bg_grad += (bg_relations * (-fg_grad[ii] / FP_num))

                # aLRP with grad formulation fg gradient
        classification_grads[fg_labels] = fg_grad
        # aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels] = relevant_bg_grad

        classification_grads /= (fg_num)

        cls_loss = 1 - prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss, rank, order

    @staticmethod
    def backward(ctx, out_grad1, out_grad2, out_grad3):
        g1, = ctx.saved_tensors
        return g1 * out_grad1, None, None, None, None


class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta=1.):
        classification_grads = torch.zeros(logits.shape).cuda()

        # Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        # Do not use bg with scores less than minimum fg logit
        # since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits) - delta

        # Get valid bg logits
        relevant_bg_labels = ((targets == 0) & (logits >= threshold_logit))
        relevant_bg_logits = logits[relevant_bg_labels]
        relevant_bg_grad = torch.zeros(len(relevant_bg_logits)).cuda()
        rank = torch.zeros(fg_num).cuda()
        prec = torch.zeros(fg_num).cuda()
        fg_grad = torch.zeros(fg_num).cuda()

        max_prec = 0
        # sort the fg logits
        order = torch.argsort(fg_logits)
        # Loops over each positive following the order
        for ii in order:
            # x_ij s as score differences with fgs
            fg_relations = fg_logits - fg_logits[ii]
            # Apply piecewise linear function and determine relations with fgs
            fg_relations = torch.clamp(fg_relations / (2 * delta) + 0.5, min=0, max=1)
            # Discard i=j in the summation in rank_pos
            fg_relations[ii] = 0

            # x_ij s as score differences with bgs
            bg_relations = relevant_bg_logits - fg_logits[ii]
            # Apply piecewise linear function and determine relations with bgs
            bg_relations = torch.clamp(bg_relations / (2 * delta) + 0.5, min=0, max=1)

            # Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos = 1 + torch.sum(fg_relations)
            FP_num = torch.sum(bg_relations)
            # Store the total since it is normalizer also for aLRP Regression error
            rank[ii] = rank_pos + FP_num

            # Compute precision for this example
            current_prec = rank_pos / rank[ii]

            # Compute interpolated AP and store gradients for relevant bg examples
            if (max_prec <= current_prec):
                max_prec = current_prec
                relevant_bg_grad += (bg_relations / rank[ii])
            else:
                relevant_bg_grad += (bg_relations / rank[ii]) * (((1 - max_prec) / (1 - current_prec)))

            # Store fg gradients
            fg_grad[ii] = -(1 - max_prec)
            prec[ii] = max_prec

            # aLRP with grad formulation fg gradient
        classification_grads[fg_labels] = fg_grad
        # aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels] = relevant_bg_grad

        classification_grads /= fg_num

        cls_loss = 1 - prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss

    @staticmethod
    def backward(ctx, out_grad1):
        g1, = ctx.saved_tensors
        return g1 * out_grad1, None, None


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria: binary cross-entropy with logits loss for classes and object
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.n_layers, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls = BCEcls
        self.BCEobj = BCEobj
        self.gr = model.gr
        self.hyp = h
        self.autobalance = autobalance
        for k in ['n_anchors', 'n_classes', 'n_layers', 'anchors', 'stride']:
            setattr(self, k, getattr(det, k))

    def __call__(self, predictions, targets):  # predictions, targets, model
        device = targets.device

        # initialize losses (classes, bounding boxes, objectness)
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_kpt = torch.zeros(1, device=device)

        targets_cls, targets_box, targets_kpt, indices, anchors = self.build_targets(predictions, targets)  # targets

        # Losses
        for i, prd_i in enumerate(predictions):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            target_obj = torch.zeros_like(prd_i[..., 0], device=device)  # target obj
            # key point loss
            if targets_kpt[i].numel() > 0:
                predicted_keypoints = prd_i[b, a, gj, gi][:, 5 + self.n_classes:]  # Get predicted keypoints
                # predicted_keypoints = prd_i[..., 5 + self.n_classes:]  # Get predicted keypoints
                loss_kpt += nn.functional.mse_loss(predicted_keypoints, targets_kpt[i])  # Calculate keypoint loss: MSE

            n_targets = b.shape[0]  # number of targets
            if n_targets:
                prd_subset = prd_i[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = prd_subset[:, :2].sigmoid() * 2. - 0.5
                pwh = (prd_subset[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, targets_box[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                loss_box += (1.0 - iou).mean()  # iou loss

                # Objectness
                target_obj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(target_obj.dtype)  # iou ratio

                # Classification loss
                if self.n_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(prd_subset[:, 5:], self.cn, device=device)  # targets
                    t[range(n_targets), targets_cls[i]] = self.cp
                    # t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    loss_cls += self.BCEcls(prd_subset[:, 5:], t)  # BCE

            obji = self.BCEobj(prd_i[..., 4], target_obj)
            loss_obj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # weight losses by hyperparameters before aggregating them
        loss_box *= self.hyp['box']
        loss_obj *= self.hyp['obj']
        loss_cls *= self.hyp['cls']
        loss_kpt *= self.hyp['kpt'] if "kpt" in self.hyp else 1.0  # weight the keypoint loss

        batch_sz = target_obj.shape[0]  # batch size

        loss = loss_box + loss_obj + loss_cls + loss_kpt
        # return loss * batch_sz, torch.cat((loss_box, loss_obj, loss_cls, loss)).detach()
        return loss * batch_sz, torch.cat((loss_box, loss_obj, loss_cls, loss_kpt, loss)).detach()

    def build_targets(self, predictions, targets):
        # check label size, expecting bounding box or bounding box + keypoints
        sz_label = targets.shape[1]
        # bounding-boxes as target:
        # (image in batch, class, x, y, w, h)
        sz_label_bbox = 6
        # keypoints as target:
        # (image in batch, class, x, y, w, h, k1x, k1y, k1visibility, ...)
        assert (sz_label >= sz_label_bbox and sz_label % 3 == 0), \
            "Unexpected shape of the targets. N0 bounding-box nor keyoints."

        idx_xy = [2, 3]  # coordinates (relative)
        idx_wh = [4, 5]  # width / height of bounding boxes
        idx_xywh = idx_xy + idx_wh

        # Build targets for compute_loss(), input targets(image, class, x, y, w, h)

        # number of anchors, targets
        n_anchors = self.n_anchors
        n_targets = targets.shape[0]

        target_cls, target_box, target_kpt, indices, all_anchors = [], [], [], [], []
        # initialize union gain (i.e. [1, 1, 1, ...])
        gain = torch.ones(sz_label + 1, device=targets.device).long()  # normalized to grid-space gain
        # initialize indices for anchors as matrix for later reshaping:
        # anchor_indices = [[0, 0, 0, ... * n_targets],
        #                   [1, 1, 1, ... * n_targets],
        #                   ... * n_anchors
        #                  ]
        anchor_indices = torch.arange(n_anchors, device=targets.device).float().view(n_anchors, 1).repeat(1, n_targets)
        # same as .repeat_interleave(n_targets)

        # append anchor_indices as element to each element of targets
        targets = torch.cat((targets.repeat(n_anchors, 1, 1), anchor_indices[:, :, None]), 2)  # append anchor indices
        # targets (bounding-box):   (batch, class, x, y, w, h, anchor) = 2 + 5
        # targets (keypoints):      (batch, class, x, y, w, h, [x1, y1, v1], ..., anchor)   = 2 + 5 + 3 * n_keypoints

        g = 0.5  # bias
        # offset to anchor points
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.n_layers):  # number of (scaling) levels / model heads in model prediction
            anchors = self.anchors[i]
            # gain: override / initialize everything related to coordinates ...
            gain[idx_xywh] = torch.tensor(predictions[i].shape)[[3, 2] * (len(idx_xywh) // 2)]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if n_targets:
                # Matches
                r = t[:, :, idx_wh] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))

                # filter anchors / matches
                t = t[j]  # filter

                # Offsets to grid cells
                gxy = t[:, idx_xy]  # grid xy
                gxi = gain[idx_xy] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((j.shape[0], 1, 1))[j]  # FIXME: was 5
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                # no targets
                t = targets[0]
                offsets = 0

            # Define
            # b, c = t[:, :2].long().T  # image, class
            gxy = t[:, idx_xy]  # grid xy
            gwh = t[:, idx_wh]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            batch = t[:, 0].long().T  # image/batch
            cls = t[:, 1].long().T  # class
            idx_anchor = t[:, -1].long()  # anchor indices
            # image/batch, best anchor, grid indices
            indices.append((batch, idx_anchor, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))

            box = torch.cat((gxy - gij, gwh), 1)
            kpts = t[:, 7:]  # batch nr., class, x, y, w, h + keypoints
            target_box.append(box)  # box
            target_cls.append(cls)  # class
            target_kpt.append(kpts)  # keypoints
            all_anchors.append(anchors[idx_anchor])  # anchors

        return target_cls, target_box, target_kpt, indices, all_anchors


class ComputeLossOTA:
    # Over-the-Air training => various different losses
    # Compute losses
    def __init__(self, model, autobalance: bool = False):
        super(ComputeLossOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria: binary cross-entropy with logits loss for classes and object
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.n_layers, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls = BCEcls
        self.BCEobj = BCEobj
        self.gr = model.gr
        self.hyp = h
        self.autobalance = autobalance
        for k in ['n_anchors', 'n_classes', 'n_layers', 'anchors', 'stride']:
            setattr(self, k, getattr(det, k))

    def __call__(self, predictions, targets, imgs):  # predictions, targets, model
        device = targets.device

        # initialize losses (classes, bounding boxes, objectness)
        loss_cls = torch.zeros(1, device=device)
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_kpt = torch.zeros(1, device=device)

        batch_sz, as_, gjs, gis, targets, anchors = self.build_targets(predictions, targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in predictions]

        # Losses
        for i, prd_i in enumerate(predictions):  # layer index, layer predictions
            b, a, gj, gi = batch_sz[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            # initialize / reset target objectness
            target_obj = torch.zeros_like(prd_i[..., 0], device=device)  # target obj
            # key point loss
            target_kpt = targets[i][:, 6:]  # get keypoints from targets
            if target_kpt.numel() > 0:
                predicted_keypoints = prd_i[b, a, gj, gi][:, 5 + self.n_classes:]  # Get predicted keypoints
                loss_kpt += nn.functional.mse_loss(predicted_keypoints, target_kpt)  # Calculate keypoint loss: MSE

            n_targets = b.shape[0]  # number of targets
            if n_targets:
                prd_subset = prd_i[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = prd_subset[:, :2].sigmoid() * 2. - 0.5
                # pxy = prd_subset[:, :2].sigmoid() * 3. - 1.
                pwh = (prd_subset[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                loss_box += (1.0 - iou).mean()  # iou loss

                # Objectness
                target_obj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(
                    target_obj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.n_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(prd_subset[:, 5:], self.cn, device=device)  # targets
                    t[range(n_targets), selected_tcls] = self.cp
                    loss_cls += self.BCEcls(prd_subset[:, 5:], t)  # BCE

            obji = self.BCEobj(prd_i[..., 4], target_obj)
            loss_obj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # weight losses by hyperparameters before aggregating them
        loss_box *= self.hyp['box']
        loss_obj *= self.hyp['obj']
        loss_cls *= self.hyp['cls']
        loss_kpt *= self.hyp['kpt'] if "kpt" in self.hyp else 1.0  # weight the keypoint loss

        loss = loss_box + loss_obj + loss_cls + loss_kpt

        batch_sz = target_obj.shape[0]  # batch size
        return loss * batch_sz, torch.cat((loss_box, loss_obj, loss_cls, loss_kpt, loss)).detach()

    def build_targets(self, predictions, targets, imgs):
        indices, anchors = self.find_3_positive(predictions, targets)
        # image/batch, best anchor, grid indices

        n_layers = len(predictions)
        # initialize
        matching_bs = [[] for _ in range(n_layers)]  # batch index
        matching_as = [[] for _ in range(n_layers)]  # best anchor index
        matching_gjs = [[] for _ in range(n_layers)]  # grid index
        matching_gis = [[] for _ in range(n_layers)]  # grid index
        matching_targets = [[] for _ in range(n_layers)]
        matching_anchors = [[] for _ in range(n_layers)]

        for batch_idx in range(predictions[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            # make coordinates absolute
            txywh = this_target[:, 2:6] * torch.tensor((imgs[batch_idx].shape[1:] * 2))
            # transform target coordinates to (absolute?) corner coordinates
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            prd_cls = []  # predictions classes
            prd_obj = []  # predictions objectness
            prd_kpt = []  # predictions keypoints
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anchors = []

            for i, prd_i in enumerate(predictions):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anchors.append(anchors[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = prd_i[b, a, gj, gi]
                prd_obj.append(fg_pred[:, 4:5])
                prd_cls.append(fg_pred[:, 5:5 + self.n_classes])
                prd_kpt.append(fg_pred[:, 5 + self.n_classes:])  # Assuming keypoints start after class prediction

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]  # / 8.
                # pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anchors[i][idx] * self.stride[i]  # / 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            prd_obj = torch.cat(prd_obj, dim=0)
            prd_cls = torch.cat(prd_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anchors = torch.cat(all_anchors, dim=0)

            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            # number of classes
            # shape: (number of targets, number of predictions, number of classes)
            gt_cls_per_image = \
                one_hot(this_target[:, 1].to(torch.int64), self.n_classes).float().unsqueeze(1).repeat(1, pxyxys.shape[0], 1)

            num_gt = this_target.shape[0]
            cls_preds_ = prd_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() \
                         * prd_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)),  # input
                gt_cls_per_image,  # target
                reduction="none"
            ).sum(-1)
            del cls_preds_  # free up memory

            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx],
                                        k=dynamic_ks[gt_idx].item(),
                                        largest=False
                                        )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks  # free up memory
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anchors = all_anchors[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(n_layers):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchors[i].append(all_anchors[layer_idx])

        for i in range(n_layers):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchors[i] = torch.cat(matching_anchors[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchors[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchors

    def find_3_positive(self, predictions: List[torch.Tensor], targets: torch.Tensor):
        # check label size, expecting bounding box or bounding box + keypoints
        sz_label = targets.shape[1]
        # bounding-boxes as target:
        # (image in batch, class, x, y, w, h)
        sz_label_bbox = 6
        # keypoints as target:
        # (image in batch, class, x, y, w, h, k1x, k1y, k1visibility, ...)
        assert (
                    sz_label >= sz_label_bbox and sz_label % 3 == 0), "Unexpected shape of the targets. N0 bounding-box nor keyoints."

        idx_xy = [2, 3]  # coordinates (relative)
        idx_wh = [4, 5]  # width / height of bounding boxes
        idx_xywh = idx_xy + idx_wh

        # Build targets for compute_loss(), input targets(image, class, x, y, w, h)
        n_anchors = self.n_anchors  # number of anchors
        n_targets = targets.shape[0]  # number of targets
        indices, anch = [], []
        # initialize union gain (i.e. [1, 1, 1, ...])
        gain = torch.ones(sz_label + 1, device=targets.device).long()  # normalized to grid-space gain

        # initialize indices for anchors as matrix for later reshaping:
        # anchor_indices = [[0, 0, 0, ... * n_targets],
        #                   [1, 1, 1, ... * n_targets],
        #                   ... * n_anchors
        #                  ]
        anchor_indices = torch.arange(n_anchors, device=targets.device).float().view(n_anchors, 1).repeat(1, n_targets)
        # same as .repeat_interleave(n_targets)

        # append anchor_indices as element to each element of targets
        targets = torch.cat((targets.repeat(n_anchors, 1, 1), anchor_indices[:, :, None]), 2)  # append anchor indices
        # targets (bounding-box):   (batch, class, x, y, w, h, anchor) = 2 + 5
        # targets (keypoints):      (batch, class, x, y, w, h, [x1, y1, v1], ..., anchor)   = 2 + 5 + 3 * n_keypoints

        g = 0.5  # bias
        # offset to anchor points
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.n_layers):  # number of (scaling) levels / model heads in model prediction 'p'
            anchors = self.anchors[i]
            # gain: override / initialize everything related to coordinates ...
            gain[idx_xywh] = torch.tensor(predictions[i].shape)[[3, 2] * (len(idx_xywh) // 2)]  # xyxy gain
            # p[0].shape = torch.Size([6, 3, 80, 80, 13]) (for bounding-boxes as well as for keypoints)
            # p[1].shape = torch.Size([6, 3, 40, 40, 13])
            # p[2].shape = torch.Size([6, 3, 20, 20, 13])

            # Match targets to anchors
            t = targets * gain
            if n_targets:
                # filter Matches by bounding boxes
                r = t[:, :, idx_wh] / anchors.repeat(1, len(idx_wh) // 2)[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(dim=2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))

                # filter anchors / matches
                t = t[j]  # filter

                # Offsets to grid cells
                gxy = t[:, idx_xy]  # grid xy
                gxi = gain[idx_xy] - gxy  # inverse
                jk = ((gxy % 1. < g) & (gxy > 1.)).T
                lm = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(jk[0]), *[el for el in jk], *[el for el in lm]))
                # assert j.shape[0] == (sz_label - 1)
                t = t.repeat((j.shape[0], 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            else:
                # no targets
                t = targets[0]
                offsets = 0

            # Define
            gxy = t[:, idx_xy]  # grid xy
            # gwh = t[:, idx_wh]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            b = t[:, 0].long().T  # image/batch,
            idx_anchor = t[:, -1].long()  # anchor indices
            # image/batch, best anchor, grid indices
            indices.append((b, idx_anchor, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            anch.append(anchors[idx_anchor])  # anchors

        return indices, anch


class ComputeLossBinOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossBinOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # MSEangle = nn.MSELoss().to(device)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.n_layers, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'n_anchors', 'n_classes', 'n_layers', 'anchors', 'stride', 'bin_count':
            setattr(self, k, getattr(det, k))

        # xy_bin_sigmoid = SigmoidBin(bin_count=11, min=-0.5, max=1.5, use_loss_regression=False).to(device)
        wh_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0, use_loss_regression=False).to(device)
        # angle_bin_sigmoid = SigmoidBin(bin_count=31, min=-1.1, max=1.1, use_loss_regression=False).to(device)
        self.wh_bin_sigmoid = wh_bin_sigmoid

    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs)
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p]

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            obj_idx = self.wh_bin_sigmoid.get_length() * 2 + 2  # x,y, w-bce, h-bce     # xy_bin_sigmoid.get_length()*2

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid

                # pxy = ps[:, :2].sigmoid() * 2. - 0.5
                ##pxy = ps[:, :2].sigmoid() * 3. - 1.
                # pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                # pbox = torch.cat((pxy, pwh), 1)  # predicted box

                # x_loss, px = xy_bin_sigmoid.training_loss(ps[..., 0:12], tbox[i][..., 0])
                # y_loss, py = xy_bin_sigmoid.training_loss(ps[..., 12:24], tbox[i][..., 1])
                w_loss, pw = self.wh_bin_sigmoid.training_loss(ps[..., 2:(3 + self.bin_count)],
                                                               selected_tbox[..., 2] / anchors[i][..., 0])
                h_loss, ph = self.wh_bin_sigmoid.training_loss(ps[..., (3 + self.bin_count):obj_idx],
                                                               selected_tbox[..., 3] / anchors[i][..., 1])

                pw *= anchors[i][..., 0]
                ph *= anchors[i][..., 1]

                px = ps[:, 0].sigmoid() * 2. - 0.5
                py = ps[:, 1].sigmoid() * 2. - 0.5

                lbox += w_loss + h_loss  # + x_loss + y_loss

                # print(f"\n px = {px.shape}, py = {py.shape}, pw = {pw.shape}, ph = {ph.shape} \n")

                pbox = torch.cat((px.unsqueeze(1), py.unsqueeze(1), pw.unsqueeze(1), ph.unsqueeze(1)), 1).to(
                    device)  # predicted box

                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.n_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, (1 + obj_idx):], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, (1 + obj_idx):], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., obj_idx], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets, imgs):
        indices, anch = self.find_3_positive(p, targets)

        matching_bs = [[] for _ in p]
        matching_as = [[] for _ in p]
        matching_gjs = [[] for _ in p]
        matching_gis = [[] for _ in p]
        matching_targets = [[] for _ in p]
        matching_anchs = [[] for _ in p]

        n_layers = len(p)

        for batch_idx in range(p[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, pi in enumerate(p):
                obj_idx = self.wh_bin_sigmoid.get_length() * 2 + 2

                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, obj_idx:(obj_idx + 1)])
                p_cls.append(fg_pred[:, (obj_idx + 1):])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]  # / 8.
                # pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i] #/ 8.
                pw = self.wh_bin_sigmoid.forward(fg_pred[..., 2:(3 + self.bin_count)].sigmoid()) * anch[i][idx][:, 0] * \
                     self.stride[i]
                ph = self.wh_bin_sigmoid.forward(fg_pred[..., (3 + self.bin_count):obj_idx].sigmoid()) * anch[i][idx][:,
                                                                                                         1] * \
                     self.stride[i]

                pxywh = torch.cat([pxy, pw.unsqueeze(1), ph.unsqueeze(1)], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                one_hot(this_target[:, 1].to(torch.int64), self.n_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                    p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)),
                gt_cls_per_image,
                reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx],
                                        k=dynamic_ks[gt_idx].item(),
                                        largest=False
                                        )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(n_layers):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(n_layers):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.n_anchors, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.n_layers):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch


class ComputeLossAuxOTA:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLossAuxOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.n_layers, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'n_anchors', 'n_classes', 'n_layers', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets, imgs):  # predictions, targets, model   
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets2(p[:self.n_layers], targets, imgs)
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p[:self.n_layers], targets, imgs)
        pre_gen_gains_aux = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.n_layers]]
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p[:self.n_layers]]

        # Losses
        for i in range(self.n_layers):  # layer index, layer predictions
            pi = p[i]
            pi_aux = p[i + self.n_layers]
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            b_aux, a_aux, gj_aux, gi_aux = bs_aux[i], as_aux_[i], gjs_aux[i], gis_aux[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            tobj_aux = torch.zeros_like(pi_aux[..., 0], device=device)  # target obj

            n_targets = b.shape[0]  # number of targets
            if n_targets:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.n_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n_targets), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            n_aux = b_aux.shape[0]  # number of targets
            if n_aux:
                ps_aux = pi_aux[b_aux, a_aux, gj_aux, gi_aux]  # prediction subset corresponding to targets
                grid_aux = torch.stack([gi_aux, gj_aux], dim=1)
                pxy_aux = ps_aux[:, :2].sigmoid() * 2. - 0.5
                # pxy_aux = ps_aux[:, :2].sigmoid() * 3. - 1.
                pwh_aux = (ps_aux[:, 2:4].sigmoid() * 2) ** 2 * anchors_aux[i]
                pbox_aux = torch.cat((pxy_aux, pwh_aux), 1)  # predicted box
                selected_tbox_aux = targets_aux[i][:, 2:6] * pre_gen_gains_aux[i]
                selected_tbox_aux[:, :2] -= grid_aux
                iou_aux = bbox_iou(pbox_aux.T, selected_tbox_aux, x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += 0.25 * (1.0 - iou_aux).mean()  # iou loss

                # Objectness
                tobj_aux[b_aux, a_aux, gj_aux, gi_aux] = (1.0 - self.gr) + self.gr * iou_aux.detach().clamp(0).type(
                    tobj_aux.dtype)  # iou ratio

                # Classification
                selected_tcls_aux = targets_aux[i][:, 1].long()
                if self.n_classes > 1:  # cls loss (only if multiple classes)
                    t_aux = torch.full_like(ps_aux[:, 5:], self.cn, device=device)  # targets
                    t_aux[range(n_aux), selected_tcls_aux] = self.cp
                    lcls += 0.25 * self.BCEcls(ps_aux[:, 5:], t_aux)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            obji_aux = self.BCEobj(pi_aux[..., 4], tobj_aux)
            lobj += obji * self.balance[i] + 0.25 * obji_aux * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets, imgs):

        indices, anch = self.find_3_positive(p, targets)

        matching_bs = [[] for _ in p]
        matching_as = [[] for _ in p]
        matching_gjs = [[] for _ in p]
        matching_gis = [[] for _ in p]
        matching_targets = [[] for _ in p]
        matching_anchs = [[] for _ in p]

        n_layers = len(p)

        for batch_idx in range(p[0].shape[0]):

            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, pi in enumerate(p):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]  # / 8.
                # pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]  # / 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                one_hot(this_target[:, 1].to(torch.int64), self.n_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                    p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(n_layers):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(n_layers):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def build_targets2(self, p, targets, imgs):

        indices, anch = self.find_5_positive(p, targets)

        matching_bs = [[] for _ in p]
        matching_as = [[] for _ in p]
        matching_gjs = [[] for _ in p]
        matching_gis = [[] for _ in p]
        matching_targets = [[] for _ in p]
        matching_anchs = [[] for _ in p]

        nl = len(p)

        for batch_idx in range(p[0].shape[0]):

            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, pi in enumerate(p):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = pi[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]  # / 8.
                # pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]  # / 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            pair_wise_iou = box_iou(txyxy, pxyxys)

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(20, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                one_hot(this_target[:, 1].to(torch.int64), self.n_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                    p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_iou_loss
            )

            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_5_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        n_anchors, n_targets = self.n_anchors, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(n_anchors, device=targets.device).float().view(n_anchors, 1).repeat(1, n_targets)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(n_anchors, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 1.0  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.n_layers):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if n_targets:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch

    def find_3_positive(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        n_anchors, n_targets = self.n_anchors, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(n_anchors, device=targets.device).float().view(n_anchors, 1).repeat(1, n_targets)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(n_anchors, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.n_layers):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if n_targets:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch
