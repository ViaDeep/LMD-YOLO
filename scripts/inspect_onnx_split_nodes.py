"""Read-only diagnostic for Split nodes relevant to NNAPI session creation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs/stage4_mobile_export_audit/onnx_fp32_640/best_deploy_fp32_source_copy.onnx"


def main() -> None:
    model = onnx.load(MODEL, load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes: dict[str, list[int | str | None]] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        dims: list[int | str | None] = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_value or dim.dim_param or None)
        shapes[value.name] = dims
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    constants: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output:
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attribute.t)

    results = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Split":
            continue
        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        axis = int(attributes.get("axis", 0))
        input_shape = shapes.get(node.input[0], [])
        split_input = None
        if len(node.input) > 1 and node.input[1]:
            value = initializers.get(node.input[1], constants.get(node.input[1]))
            if value is not None:
                split_input = np.asarray(value).reshape(-1).astype(np.int64).tolist()
        results.append({
            "graph_index": index,
            "name": node.name,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "output_count": len(node.output),
            "axis": axis,
            "input_shape": input_shape,
            "axis_dimension": input_shape[axis] if input_shape and -len(input_shape) <= axis < len(input_shape) else None,
            "split_input": split_input,
            "num_outputs_attribute": attributes.get("num_outputs"),
        })
    print(json.dumps({"opset": model.opset_import[0].version, "split_nodes": results}, indent=2))


if __name__ == "__main__":
    main()
