import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import math

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from utils.loss import SigmoidBin

from typing import Union

try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, n_classes: int = 80, anchors=(), ch=()):  # detection layer
        super().__init__()

        self.n_classes = n_classes  # number of classes
        self.n_out = n_classes + 5  # number of outputs per anchor
        self.n_layers = len(anchors)  # number of detection layers
        self.n_anchors = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.n_layers  # init grid
        a = torch.tensor(anchors).float().view(self.n_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.n_layers, 1, -1, 1, 1, 2))  # shape(nl, 1, na, 1, 1, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.n_out * self.n_anchors, 1) for x in ch)  # output conv

    def forward(self, x):
        # map_attributes = {
        #     "n_layers": "nl",
        #     "n_anchors": "na",
        #     "n_classes": "nc"
        # }
        # for attr_long, attr_short in map_attributes.items():
        #     if hasattr(self, attr_short):
        #         setattr(self, attr_long, getattr(self, attr_short))
        # if not hasattr(self, "n_out"):
        #     self.n_out = self.n_classes + 5

        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.n_layers):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.n_anchors, self.n_out, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.n_classes + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.n_out))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    @staticmethod
    def convert(z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0],
                                       [0, 1, 0, 1],
                                       [-0.5, 0, 0.5, 0],
                                       [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix                          
        return box, score


class IDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, n_classes: int = 80, anchors=(), channels: torch.Tensor = ()):  # detection layer
        super().__init__()
        self.n_classes = n_classes  # number of classes
        self.n_out = n_classes + 5  # number of outputs per anchor
        self.n_layers = len(anchors)  # number of detection layers
        self.n_anchors = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.n_layers  # init grid
        a = torch.tensor(anchors).float().view(self.n_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.n_layers, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.n_out * self.n_anchors, 1) for x in channels)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in channels)
        self.im = nn.ModuleList(ImplicitM(self.n_out * self.n_anchors) for _ in channels)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.n_layers):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            # x[i] = self.im[i](x[i])  # FIXME: lines does not exist in segmentation code
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.n_anchors, self.n_out, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.n_out))

        return x if self.training else (torch.cat(z, 1), x)
    
    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.n_layers):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.n_anchors, self.n_out, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.n_classes + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.n_out))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)            
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2), self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)
            
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    @staticmethod
    def convert(z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0],
                                       [0, 1, 0, 1],
                                       [-0.5, 0, 0.5, 0],
                                       [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix                          
        return box, score


class IKeypoint(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc: int = 80, anchors=(), nkpt: int = 17, ch=(), inplace: bool = True, dw_conv_kpt: bool = False):  # detection layer
        super().__init__()
        self.n_classes = nc  # number of classes | usually 1
        self.n_kpt = nkpt
        self.dw_conv_kpt = dw_conv_kpt
        self.n_out_det = nc + 5  # number of outputs per anchor for box and class
        self.n_out_kpt = 3 * self.n_kpt  # number of outputs per anchor for keypoints
        self.n_out = self.n_out_det + self.n_out_kpt
        self.n_layers = len(anchors)  # number of detection layers
        self.n_anchors = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.n_layers  # init grid
        self.flip_test = False
        a = torch.tensor(anchors).float().view(self.n_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.n_layers, 1, -1, 1, 1, 2))  # shape(nl, 1, na, 1, 1, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.n_out_det * self.n_anchors, 1) for x in ch)  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.n_out_det * self.n_anchors) for _ in ch)
        
        if self.n_kpt is not None:
            if self.dw_conv_kpt:  # keypoint head is slightly more complex
                self.m_kpt = nn.ModuleList(
                    nn.Sequential(DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), nn.Conv2d(x, self.n_out_kpt * self.n_anchors, 1)) for x in ch)
            else:  # keypoint head is a single convolution
                self.m_kpt = nn.ModuleList(nn.Conv2d(x, self.n_out_kpt * self.n_anchors, 1) for x in ch)

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.n_layers):
            if self.n_kpt is None or self.n_kpt <= 0:
                x[i] = self.im[i](self.m[i](self.ia[i](x[i])))  # conv
            else:
                x[i] = torch.cat((self.im[i](self.m[i](self.ia[i](x[i]))), self.m_kpt[i](x[i])), axis=1)

            batch_size, _, ny, nx = x[i].shape  # x(batch_size, 255, 20, 20) to x(batch_size, 3, 20, 20, 85) # torch.Size([1, 39, 32, 32])
            x[i] = x[i].view(batch_size, self.n_anchors, self.n_out, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_det = x[i][..., :self.n_out_det]
            x_kpt = x[i][..., self.n_out_det:]
            assert x_det.shape[-1] == self.n_out_det
            assert x_kpt.shape[-1] == self.n_out_kpt

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.n_kpt == 0:
                    y = x[i].sigmoid()
                else:
                    y = x_det.sigmoid()

                if self.inplace:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.n_anchors, 1, 1, 2)  # wh
                    if self.n_kpt > 0:  # TODO: Fix here
                        # INFO: x_kpt.shape = (batch_size, num_anchors, num_boxes, num_keypoints)
                        # INFO: x_kpt[..., 0::3] select every 3rd element, i.e. select every x value
                        # INFO: kpt_grid_x.repeat(1, 1, 1, 1, self.n_kpt) = (batch_size, num_anchors, num_boxes, num_keypoints)
                        x_kpt[..., 0::3] = (x_kpt[..., 0::3] * 2. - 0.5 + kpt_grid_x.repeat(1, 1, 1, 1, self.n_kpt)) * self.stride[i]  # x
                        # INFO: x_kpt[..., 1::3] select every 3rd element starting at the 2nd element, i.e. select every y value
                        x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1, 1, 1, 1, self.n_kpt)) * self.stride[i]  # y
                        #x_kpt[..., 0::3] = (x_kpt[..., ::3] + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        #x_kpt[..., 1::3] = (x_kpt[..., 1::3] + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        #print('=============')
                        #print(self.anchor_grid[i].shape)
                        #print(self.anchor_grid[i][...,0].unsqueeze(4).shape)
                        #print(x_kpt[..., 0::3].shape)
                        #x_kpt[..., 0::3] = ((x_kpt[..., 0::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        #x_kpt[..., 1::3] = ((x_kpt[..., 1::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        #x_kpt[..., 0::3] = (((x_kpt[..., 0::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        #x_kpt[..., 1::3] = (((x_kpt[..., 1::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                    y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim=-1)

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    if self.n_kpt != 0:
                        y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[i].repeat((1, 1, 1, 1, self.n_kpt))) * self.stride[i]  # xy
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(batch_size, -1, self.n_out))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, n_classes=80, anchors=(), n_masks=32, n_protos=256, ch=()):
        super().__init__(n_classes, anchors, ch)
        self.n_masks = n_masks  # number of masks
        self.n_protos = n_protos  # number of protos
        self.n_out = 5 + n_classes + self.n_masks  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.n_out * self.n_anchors, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.n_protos, self.n_masks)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], (x[1], p))


class ISegment(IDetect):
    # YOLOR Segment head for segmentation models
    def __init__(self, n_classes: int = 80, anchors: tuple = (), n_masks: int = 32, n_protos: int = 256, ch=()):
        super().__init__(n_classes, anchors, ch)
        self.n_masks = n_masks  # number of masks
        self.n_protos = n_protos  # number of protos
        self.n_out = 5 + n_classes + self.n_masks  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.n_out * self.n_anchors, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.n_protos, self.n_masks)  # protos
        self.detect = IDetect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], (x[1], p))


class IRSegment(IDetect):
    # YOLOR Segment head for segmentation models
    def __init__(self, n_classes=80, anchors=(), n_masks=32, npr=256, ch=()):
        super().__init__(n_classes, anchors, ch)
        self.n_masks = n_masks  # number of masks
        self.n_protos = npr  # number of protos
        self.n_out = 5 + n_classes + self.n_masks  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.n_out * self.n_anchors, 1) for x in ch[self.nl:])  # output conv
        self.refine = Refine(ch[:self.n_layers], self.n_protos, self.n_masks)  # protos
        self.detect = IDetect.forward

    def forward(self, x):
        p = self.refine(x[:self.nl])
        x = self.detect(self, x[self.n_layers:])
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], (x[1], p))


class MT(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, n_classes: int = 80, anchors=(), attn=None, mask_iou: bool = False, ch=()):  # detection layer
        super().__init__()
        self.n_classes = n_classes  # number of classes
        self.n_out = n_classes + 5  # number of outputs per anchor
        self.n_layers = len(anchors)  # number of detection layers
        self.n_anchors = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.n_layers  # init grid
        a = torch.tensor(anchors).float().view(self.n_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl, na, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.n_layers, 1, -1, 1, 1, 2))  # shape(nl, 1, na, 1, 1, 2)
        self.original_anchors = anchors
        self.m = nn.ModuleList(nn.Conv2d(x, self.n_out * self.n_anchors, 1) for x in ch[0])  # output conv
        if mask_iou:
            self.m_iou = nn.ModuleList(nn.Conv2d(x, self.n_anchors, 1) for x in ch[0])  # output con
        self.mask_iou = mask_iou
        self.attn = attn
        if attn is not None:
            # self.attn_m = nn.ModuleList(nn.Conv2d(x, attn * self.n_anchors, 3, padding=1) for x in ch)  # output conv
            self.attn_m = nn.ModuleList(nn.Conv2d(x, attn * self.n_anchors, 1) for x in ch[0])  # output conv
            # self.attn_m = nn.ModuleList(nn.Conv2d(x, attn * self.n_anchors,  kernel_size=3, stride=1, padding=1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        za = []
        zi = []
        attn = [None] * self.n_layers
        iou = [None] * self.n_layers
        self.training |= self.export
        output = dict()
        for i in range(self.n_layers):
            if self.attn is not None:
                attn[i] = self.attn_m[i](x[0][i])  # conv
                bs, _, ny, nx = attn[i].shape  # x(bs,2352,20,20) to x(bs,3,20,20,784)
                attn[i] = attn[i].view(bs, self.n_anchors, self.attn, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if self.mask_iou:
                iou[i] = self.m_iou[i](x[0][i])
            x[0][i] = self.m[i](x[0][i])  # conv

            bs, _, ny, nx = x[0][i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[0][i] = x[0][i].view(bs, self.n_anchors, self.n_out, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if self.mask_iou:
                iou[i] = iou[i].view(bs, self.n_anchors, ny, nx).contiguous()

            if not self.training:  # inference
                za.append(attn[i].view(bs, -1, self.attn))
                if self.mask_iou:
                    zi.append(iou[i].view(bs, -1))
                if self.grid[i].shape[2:4] != x[0][i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[0][i].device)

                y = x[0][i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 3. - 1.0 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.n_out))
        output["mask_iou"] = None
        if not self.training:
            output["test"] = torch.cat(z, 1)
            if self.attn is not None:
                output["attn"] = torch.cat(za, 1)
            if self.mask_iou:
                output["mask_iou"] = torch.cat(zi, 1).sigmoid()
        else:
            if self.attn is not None:
                output["attn"] = attn
            if self.mask_iou:
                output["mask_iou"] = iou
        output["bbox_and_cls"] = x[0]
        output["bases"] = x[1]
        output["sem"] = x[2]

        return output

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class IAuxDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, n_classes=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.n_classes = n_classes  # number of classes
        self.n_out = n_classes + 5  # number of outputs per anchor
        self.n_layers = len(anchors)  # number of detection layers
        self.n_anchors = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.n_layers  # init grid
        a = torch.tensor(anchors).float().view(self.n_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.n_layers, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.n_out * self.n_anchors, 1) for x in ch[:self.n_layers])  # output conv
        self.m2 = nn.ModuleList(nn.Conv2d(x, self.n_out * self.n_anchors, 1) for x in ch[self.n_layers:])  # output conv
        
        self.ia = nn.ModuleList(ImplicitA(x) for x in ch[:self.n_layers])
        self.im = nn.ModuleList(ImplicitM(self.n_out * self.n_anchors) for _ in ch[:self.n_layers])

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.n_layers):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.n_anchors, self.n_out, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            x[i + self.n_layers] = self.m2[i](x[i + self.n_layers])
            x[i + self.n_layers] = x[i + self.n_layers].view(bs, self.n_anchors, self.n_out, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.n_classes + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.n_out))

        return x if self.training else (torch.cat(z, 1), x[:self.n_layers])

    def fuseforward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.n_layers):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.n_anchors, self.n_out, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].data  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.n_out))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)            
        else:
            out = (torch.cat(z, 1), x)

        return out
    
    def fuse(self):
        print("IAuxDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2), self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    @staticmethod
    def convert(z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0],
                                       [0, 1, 0, 1],
                                       [-0.5, 0, 0.5, 0],
                                       [0, -0.5, 0, 0.5]],
                                      dtype=torch.float32,
                                      device=z.device)
        box @= convert_matrix                          
        return box, score


class IBin(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, n_classes: int = 80, anchors=(), ch=(), bin_count: int = 21):  # detection layer
        super().__init__()
        self.n_classes = n_classes  # number of classes
        self.bin_count = bin_count

        self.w_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        self.h_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        # classes, x,y,obj
        self.n_out = n_classes + 3 + \
                     self.w_bin_sigmoid.get_length() + self.h_bin_sigmoid.get_length()   # w-bce, h-bce
            # + self.x_bin_sigmoid.get_length() + self.y_bin_sigmoid.get_length()

        self.n_layers = len(anchors)  # number of detection layers
        self.n_anchors = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.n_layers  # init grid
        a = torch.tensor(anchors).float().view(self.n_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.n_layers, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.n_out * self.n_anchors, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.n_out * self.n_anchors) for _ in ch)

    def forward(self, x):

        #self.x_bin_sigmoid.use_fw_regression = True
        #self.y_bin_sigmoid.use_fw_regression = True
        self.w_bin_sigmoid.use_fw_regression = True
        self.h_bin_sigmoid.use_fw_regression = True

        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.n_layers):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.n_anchors, self.n_out, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh


                #px = (self.x_bin_sigmoid.forward(y[..., 0:12]) + self.grid[i][..., 0]) * self.stride[i]
                #py = (self.y_bin_sigmoid.forward(y[..., 12:24]) + self.grid[i][..., 1]) * self.stride[i]

                pw = self.w_bin_sigmoid.forward(y[..., 2:24]) * self.anchor_grid[i][..., 0]
                ph = self.h_bin_sigmoid.forward(y[..., 24:46]) * self.anchor_grid[i][..., 1]

                #y[..., 0] = px
                #y[..., 1] = py
                y[..., 2] = pw
                y[..., 3] = ph

                y = torch.cat((y[..., 0:4], y[..., 46:]), dim=-1)

                z.append(y.view(bs, -1, y.shape[-1]))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(
            self,
            cfg: Union[str, Path] = 'yolor-csp-c.yaml',
            ch: int = 3,
            nc: int = None,
            anchors=None,
            nkpt: int = None
    ):  # model, input channels, number of classes
        super().__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name

            with open(cfg, "r") as fid:
                self.yaml = yaml.load(fid, Loader=yaml.SafeLoader)  # model dict

        # Define model (assign to YAML config and as local variable)
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        nc = self.yaml['nc'] = self.yaml.get('nc', nc)
        nkpt = self.yaml['nkpt'] = self.yaml.get('nkpt', nkpt)

        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, IDetect, IAuxDetect, IBin, IKeypoint, MT, Segment, ISegment, IRSegment)):
            s = 256  # 2x min stride
            if isinstance(m, IAuxDetect):
                forward = lambda x: self.forward(x)[:4]
                # m.stride = torch.tensor(
                #     [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))[:4]])  # forward
            # elif isinstance(m, MT):
            #     temp = self.forward(torch.zeros(1, ch, s, s))
            #     if isinstance(temp, list):
            #         temp = temp[0]
            #     m.stride = torch.tensor([s / x.shape[-2] for x in temp["bbox_and_cls"]])  # forward
            elif isinstance(m, (Segment, ISegment, IRSegment)):
                forward = lambda x: self.forward(x)[0]
                # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])[0]  # forward
            else:
                forward = self.forward
                # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward

            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride

            if isinstance(m, IAuxDetect):
                self._initialize_aux_biases()  # only run once
            elif isinstance(m, IBin):
                self._initialize_biases_bin()  # only run once
            elif isinstance(m, IKeypoint):
                self._initialize_biases_kpt()  # only run once
            else:  # Detect, IDetect, MT, Segments
                self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment: bool = False, profile: bool = False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced = False

            if self.traced:
                if isinstance(m, (Detect, IDetect, IAuxDetect, IKeypoint, ISegment)):
                    break

            if profile:
                c = isinstance(m, (Detect, IDetect, IAuxDetect, IBin))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.n_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.n_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_aux_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, mi2, s in zip(m.m, m.m2, m.stride):  # from
            b = mi.bias.view(m.n_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.n_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            b2 = mi2.bias.view(m.n_anchors, -1)  # conv.bias(255) to (3,85)
            b2.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b2.data[:, 5:] += math.log(0.6 / (m.n_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi2.bias = torch.nn.Parameter(b2.view(-1), requires_grad=True)

    def _initialize_biases_bin(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Bin() module
        bc = m.bin_count
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.n_anchors, -1)  # conv.bias(255) to (3,85)
            old = b[:, (0, 1, 2, bc + 3)].data
            obj_idx = 2 * bc + 4
            b[:, :obj_idx].data += math.log(0.6 / (bc + 1 - 0.99))
            b[:, obj_idx].data += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, (obj_idx + 1):].data += math.log(0.6 / (m.n_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            b[:, (0, 1, 2, bc + 3)].data = old
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _initialize_biases_kpt(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        mdl = self.model[-1]  # Detect() module
        for mi, s in zip(mdl.m, mdl.stride):  # from
            b = mi.bias.view(mdl.n_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (mdl.n_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.n_anchors, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ')
        for m in self.model.modules():
            if isinstance(m, RepConv):
                #print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
            elif isinstance(m, RepConv_OREPA):
                #print(f" switch_to_deploy")
                m.switch_to_deploy()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
            elif isinstance(m, (IDetect, IAuxDetect)):
                m.fuse()
                m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


class SegmentationModel(Model):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


def parse_model(data_dict, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    # do not rename variables due to eval() function!
    anchors = data_dict['anchors']
    nc = data_dict['nc']  # number of classes
    gd = data_dict['depth_multiple']  # gain depth
    gw = data_dict['width_multiple']  # gain width
    nkpt = data_dict['nkpt'] if 'nkpt' in data_dict else 'nkpt'  # number of keypoints
    n_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    n_out = n_anchors * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (from_layer, n_filter, module, args) in enumerate(data_dict['backbone'] + data_dict['head']):  # from, number, module, args
        module = eval(module) if isinstance(module, str) else module  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n_filter = max(round(n_filter * gd), 1) if n_filter > 1 else n_filter  # depth gain
        if module in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC,
                      SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv,
                      Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                      RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                      Res, ResCSPA, ResCSPB, ResCSPC,
                      RepRes, RepResCSPA, RepResCSPB, RepResCSPC,
                      ResX, ResXCSPA, ResXCSPB, ResXCSPC,
                      RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC,
                      Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                      SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                      SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC]:
            c1, c2 = ch[from_layer], args[0]
            if c2 != n_out:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if module in [DownC, SPPCSPC, GhostSPPCSPC,
                          BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
                          RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,
                          ResCSPA, ResCSPB, ResCSPC,
                          RepResCSPA, RepResCSPB, RepResCSPC,
                          ResXCSPA, ResXCSPB, ResXCSPC,
                          RepResXCSPA, RepResXCSPB, RepResXCSPC,
                          GhostCSPA, GhostCSPB, GhostCSPC,
                          STCSPA, STCSPB, STCSPC,
                          ST2CSPA, ST2CSPB, ST2CSPC]:
                args.insert(2, n_filter)  # number of repeats
                n_filter = 1
        elif module is nn.BatchNorm2d:
            args = [ch[from_layer]]
        elif module is Concat:
            c2 = sum([ch[x] for x in from_layer])
        elif module is Chuncat:
            c2 = sum([ch[x] for x in from_layer])
        elif module is Shortcut:
            c2 = ch[from_layer[0]]
        elif module is Foldcut:
            c2 = ch[from_layer] // 2
        elif module in [Detect, IDetect, IAuxDetect, IBin, IKeypoint, Segment, ISegment, IRSegment]:
            args.append([ch[x] for x in from_layer])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(from_layer)
        elif module is ReOrg:
            c2 = ch[from_layer] * 4
        elif module in [Merge]:
            c2 = args[0]
        elif module in [MT]:
            if len(args) == 3:
                args.append(False)
            args.append([ch[x] for x in from_layer])
        elif module is Contract:
            c2 = ch[from_layer] * args[0] ** 2
        elif module is Expand:
            c2 = ch[from_layer] // args[0] ** 2
        elif module is Refine:
            args.append([ch[x] for x in from_layer])
            c2 = args[0]
        else:
            c2 = ch[from_layer]

        m_ = nn.Sequential(*[module(*args) for _ in range(n_filter)]) if n_filter > 1 else module(*args)  # module
        t = str(module)[8:-2].replace('__main__.', '')  # module type
        n_para = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, from_layer, t, n_para  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, from_layer, n_filter, n_para, t, args))  # print
        save.extend(x % i for x in ([from_layer] if isinstance(from_layer, int) else from_layer) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolor-csp-c.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()
    
    if opt.profile:
        img = torch.rand(1, 3, 640, 640).to(device)
        y = model(img, profile=True)

