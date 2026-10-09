#!/usr/bin/env python3
"""Inspect one hash-pinned local ONNX graph without inference or sidecar reads."""

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import onnx


def protobuf_messages(message, path="model"):
    """Visit every present message, including functions, training and sparse data."""
    yield path, message
    for field, value in message.ListFields():
        if field.type != field.TYPE_MESSAGE:
            continue
        if field.is_repeated:
            for index, child in enumerate(value):
                yield from protobuf_messages(child, f"{path}.{field.name}[{index}]")
        else:
            yield from protobuf_messages(value, f"{path}.{field.name}")


def declared_type(value):
    kind = value.WhichOneof("value")
    if kind is None:
        return {"kind": "unknown"}
    detail = getattr(value, kind)
    result = {"kind": kind}
    if kind in ("tensor_type", "sparse_tensor_type"):
        result["element_type"] = onnx.TensorProto.DataType.Name(detail.elem_type)
        result["shape"] = None
        if detail.HasField("shape"):
            result["shape"] = []
            for dim in detail.shape.dim:
                dim_kind = dim.WhichOneof("value")
                result["shape"].append(
                    {"kind": "static", "value": dim.dim_value} if dim_kind == "dim_value"
                    else {"kind": "dynamic", "parameter": dim.dim_param} if dim_kind == "dim_param"
                    else {"kind": "unknown"}
                )
    elif kind in ("sequence_type", "optional_type"):
        result["element_type"] = declared_type(detail.elem_type)
    elif kind == "map_type":
        result.update(key_type=onnx.TensorProto.DataType.Name(detail.key_type),
                      value_type=declared_type(detail.value_type))
    elif kind == "opaque_type":
        result.update(domain=detail.domain, name=detail.name)
    return result


def inspect_model(path, expected_sha256, max_bytes):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected SHA-256 must contain exactly 64 hexadecimal characters")
    if max_bytes <= 0:
        raise ValueError("max-bytes must be positive")
    digest, payload, size = hashlib.sha256(), io.BytesIO(), 0
    with Path(path).open("rb") as source:
        if os.fstat(source.fileno()).st_size > max_bytes:
            raise ValueError("model exceeds max-bytes")
        while chunk := source.read(min(1024 * 1024, max_bytes - size + 1)):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("model exceeds max-bytes")
            digest.update(chunk)
            payload.write(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256.lower():
        raise ValueError("model SHA-256 does not match expected-sha256")
    payload.seek(0)
    model = onnx.load_model(payload, format="protobuf", load_external_data=False)
    external, counts, functions = [], Counter(), []
    for location, message in protobuf_messages(model):
        proto_name = message.DESCRIPTOR.name
        if proto_name == "TensorProto" and (
            message.data_location == onnx.TensorProto.EXTERNAL or message.external_data
        ):
            external.append({"path": location, "tensor_name": message.name,
                             "data_location": onnx.TensorProto.DataLocation.Name(message.data_location),
                             "references": [{"key": item.key, "value": item.value}
                                            for item in message.external_data]})
        elif proto_name == "NodeProto":
            counts[(message.domain, message.op_type)] += 1
        elif proto_name == "FunctionProto":
            functions.append({"path": location, "domain": message.domain,
                              "name": message.name, "overload": message.overload,
                              "inputs": list(message.input), "outputs": list(message.output),
                              "opsets": [{"domain": op.domain, "version": op.version}
                                         for op in message.opset_import],
                              "node_count": len(message.node)})
    if not external:
        onnx.checker.check_model(model, full_check=False)
    graph = model.graph
    return {
        "status": "blocked_external_data" if external else "embedded_graph_checked",
        "artifact": {"path": str(Path(path).resolve()), "bytes": size, "sha256": actual_sha256},
        "onnx_version": onnx.__version__, "ir_version": model.ir_version,
        "opsets": [{"domain": op.domain, "version": op.version} for op in model.opset_import],
        "graph": {"name": graph.name,
                  "inputs": [{"name": item.name, "declared_type": declared_type(item.type)}
                             for item in graph.input],
                  "outputs": [{"name": item.name, "declared_type": declared_type(item.type)}
                              for item in graph.output]},
        "node_counts_scope": "all graphs, functions and training messages in ModelProto",
        "node_counts": [{"domain": domain, "op_type": op, "count": count}
                        for (domain, op), count in sorted(counts.items())],
        "local_functions": functions, "external_data": external,
        "direct_input_consumers": {
            item.name: [{"index": index, "name": node.name, "domain": node.domain,
                         "op_type": node.op_type,
                         "input_positions": [i for i, name in enumerate(node.input) if name == item.name]}
                        for index, node in enumerate(graph.node) if item.name in node.input]
            for item in graph.input
        },
        "verification": {
            "structural_checker": "skipped_external_data" if external else "passed_embedded_model",
            "full_check": False, "custom_domain_schemas_checked": False,
            "normalization": "unverified", "runtime_compatibility": "unverified",
            "inference": "unverified", "accuracy": "unverified",
            "scope": "declared graph facts only; no sidecar reads, shape inference or weight dump",
        },
    }


def write_report(output, report):
    staged = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", delete=False) as temporary:
            staged = Path(temporary.name)
            json.dump(report, temporary, indent=2)
            temporary.write("\n")
        os.replace(staged, output)
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--max-bytes", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.model.resolve() == args.output.resolve():
            raise ValueError("output must differ from the input model")
        report = inspect_model(args.model, args.expected_sha256, args.max_bytes)
        write_report(args.output, report)
    except Exception as error:
        print(f"ONNX inspection failed: {type(error).__name__}: {str(error)[:500]}", file=sys.stderr)
        return 1
    print(f"{report['status']}: {args.output}")
    return 2 if report["external_data"] else 0


if __name__ == "__main__":
    sys.exit(main())
