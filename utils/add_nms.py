import numpy as np
import onnx
from onnx import helper, shape_inference

import logging

LOGGER = logging.getLogger(__name__)


class RegisterNMS(object):
    model = None

    def __init__(self, onnx_model_path: str, precision: str = "fp32"):
        self.model_path = onnx_model_path
        self.batch_size = 1

    def save(self, output_path):
        if self.model is not None:
            onnx.save(self.model, output_path)
            LOGGER.info(f"Saved ONNX model to {output_path}")
        else:
            raise Exception("Model was not registered yet. Please call register_model() first.")

    def register_nms(
            self,
            score_thresh: float = 0.25,
            nms_iou_thresh: float = 0.45,
            detections_per_img: int = 100
    ):
        opset_version_act = onnx.defs.onnx_opset_version()
        if opset_version_act < 13:
            raise Exception(
                f"Please upgrade to ONNX opset version 13 or higher to use NonMaxSuppression. "
                f"Current version is {opset_version_act}."
            )
        onnx_model = onnx.load(self.model_path)
        # model output = non-max-suppression input. Shape: (-1 x 7) [batch_index, x1, y1, x2?, y2?, class_id, conf]

        # Get the output names
        output_names = [output.name for output in onnx_model.graph.output]

        # Slice the tensors
        # ---- bounding box
        bboxes_start_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['bboxes_start_constant'],
            value=helper.make_tensor(name="starts_tensor", data_type=onnx.TensorProto.INT32, dims=[1], vals=[1])
        )

        bboxes_end_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['bboxes_end_constant'],
            value=helper.make_tensor(name="ends_tensor", data_type=onnx.TensorProto.INT32, dims=[1], vals=[4])
        )

        # Create a Slice node to extract elements from the first to the fourth column
        bboxes = helper.make_node(
            'Slice',  # Node type
            inputs=[output_names[0], 'bboxes_start_constant', 'bboxes_end_constant'],  # Input tensors
            outputs=['bboxes'],  # Output tensor name
        )
        # Add the Slice node to the graph
        print("extend node to slice box coordinates ...")
        onnx_model.graph.node.extend([bboxes_start_constant, bboxes_end_constant, bboxes])
        onnx.checker.check_model(onnx_model)  # check onnx model

        # ---- scores
        scores_start_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['scores_start_constant'],
            value=helper.make_tensor(name="starts_tensor", data_type=onnx.TensorProto.INT32, dims=[1], vals=[5])
        )

        scores_end_constant = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['scores_end_constant'],
            value=helper.make_tensor(name="ends_tensor", data_type=onnx.TensorProto.INT32, dims=[1], vals=[6])
        )

        # Create a Slice node to extract elements from the first to the fourth column
        scores = helper.make_node(
            'Slice',  # Node type
            inputs=[output_names[0], 'scores_start_constant', 'scores_end_constant'],  # Input tensors
            outputs=['scores'],  # Output tensor name
        )

        # Add the Slice node to the graph
        print("extend node to slice score vector ...")
        onnx_model.graph.node.extend([scores_start_constant, scores_end_constant, scores])
        onnx.checker.check_model(onnx_model)  # check onnx model

        # ---- non-maximum-suppression
        # input constants
        simple_constants = [
            # Maximum number of output boxes to keep per class
            ("max_output_boxes_per_class", detections_per_img, onnx.TensorProto.INT64),
            # IoU threshold for NMS
            ("iou_threshold", nms_iou_thresh, onnx.TensorProto.FLOAT),
            # Score threshold for NMS
            ("score_threshold ", score_thresh, onnx.TensorProto.FLOAT)
        ]

        for ky, vl, ty in simple_constants:
            constant = onnx.helper.make_tensor(ky, ty, [1], [vl])
            onnx_model.graph.initializer.append(constant)

        # create node with non-maximum-suppression
        nms_node = helper.make_node(
            op_type="NonMaxSuppression",
            inputs=["bboxes", "scores"] + [el[0] for el in simple_constants],
            outputs=["selected_indices"],
            name="nms_node",
            # attribute=list(attributes.values())
        )
        # adjust attribute(s)
        # Specify box format: 0 = [x1, y1, x2, y2] 1 = [x_center, y_center, width, height]
        center_point_box = onnx.helper.make_attribute("center_point_box", 0, attr_type=onnx.AttributeProto.INT)
        nms_node.attribute.append(center_point_box)

        # Add the NMS node to the model
        print("append node with nms ...")
        onnx_model.graph.node.append(nms_node)
        # onnx_model.graph.output.extend(op_outputs)
        onnx.checker.check_model(onnx_model)  # check onnx model

        print("infer shape ...")
        # Infer shapes after modifying the graph
        # inferred_model = shape_inference.infer_shapes(onnx_model)

        # Save the updated model
        # self.save(model, "updated_model.onnx")
        # onnx.checker.check_model(inferred_model)  # check onnx model

        # Store the updated model
        self.model = onnx_model

        LOGGER.info(f"Created NMS plugin 'NonMaxSuppression'.")

        return onnx_model

