"""Tiny generated ONNX fixtures test the tool, not NVIDIA model behavior."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import onnx
from onnx import helper, TensorProto


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect-onnx-model.py"
SPEC = importlib.util.spec_from_file_location("onnx_inspection", SCRIPT)
inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspector)


def tiny_model(shape=(1,)):
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"], name="identity")],
        "generated-tool-fixture",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def external_tensor(name):
    tensor = TensorProto(name=name, data_type=TensorProto.FLOAT, dims=[1],
                         data_location=TensorProto.EXTERNAL)
    tensor.external_data.add(key="location", value="absent-sidecar.bin")
    return tensor


class OnnxInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_path = self.root / "fixture.onnx"
        self.output = self.root / "report.json"

    def save(self, model):
        payload = model.SerializeToString()
        self.model_path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest(), len(payload)

    def cli(self, digest, max_bytes):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.model_path), "--expected-sha256", digest,
             "--max-bytes", str(max_bytes), "--output", str(self.output)],
            capture_output=True, text=True,
        )

    def test_declared_shapes_and_consumer_report(self):
        digest, size = self.save(tiny_model((0, "batch", None)))
        with patch.object(onnx, "load_model", wraps=onnx.load_model) as loader:
            report = inspector.inspect_model(self.model_path, digest, size)
        self.assertIs(loader.call_args.kwargs["load_external_data"], False)
        self.assertEqual(report["status"], "embedded_graph_checked")
        self.assertEqual(report["graph"]["inputs"][0]["declared_type"], {
            "kind": "tensor_type", "element_type": "FLOAT",
            "shape": [{"kind": "static", "value": 0},
                      {"kind": "dynamic", "parameter": "batch"}, {"kind": "unknown"}],
        })
        self.assertEqual(report["direct_input_consumers"]["input"], [
            {"index": 0, "name": "identity", "domain": "", "op_type": "Identity",
             "input_positions": [0]}])
        self.assertEqual(report["node_counts"], [{"domain": "", "op_type": "Identity", "count": 1}])
        for fact in ("normalization", "runtime_compatibility", "inference", "accuracy"):
            self.assertEqual(report["verification"][fact], "unverified")

    def test_scalar_and_absent_shape_are_distinct(self):
        scalar = helper.make_tensor_type_proto(TensorProto.FLOAT, [])
        absent = helper.make_tensor_type_proto(TensorProto.FLOAT, None)
        self.assertEqual(inspector.declared_type(scalar)["shape"], [])
        self.assertIsNone(inspector.declared_type(absent)["shape"])

    def test_hash_and_size_fail_before_parser(self):
        digest, size = self.save(tiny_model())
        with patch.object(onnx, "load_model", side_effect=AssertionError("parser reached")) as loader:
            with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
                inspector.inspect_model(self.model_path, "0" * 64, size)
            with self.assertRaisesRegex(ValueError, "exceeds max-bytes"):
                inspector.inspect_model(self.model_path, digest, size - 1)
            loader.assert_not_called()

    def test_failures_preserve_existing_output(self):
        self.output.write_text("existing report\n")
        digest, size = self.save(tiny_model())
        cases = [("0" * 64, size), (digest, size - 1)]
        for expected, limit in cases:
            with self.subTest(expected=expected, limit=limit):
                self.assertNotEqual(self.cli(expected, limit).returncode, 0)
                self.assertEqual(self.output.read_text(), "existing report\n")
        self.model_path.write_bytes(b"not a protobuf model")
        digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self.assertNotEqual(self.cli(digest, 100).returncode, 0)
        self.assertEqual(self.output.read_text(), "existing report\n")
        self.model_path.unlink()
        self.assertNotEqual(self.cli(digest, 100).returncode, 0)
        self.assertEqual(self.output.read_text(), "existing report\n")

    def test_malformed_embedded_graph_is_rejected(self):
        model = tiny_model()
        model.graph.node[0].input[0] = "undefined-value"
        digest, size = self.save(model)
        self.output.write_text("existing report")
        result = self.cli(digest, size)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ValidationError", result.stderr)
        self.assertEqual(self.output.read_text(), "existing report")

    def test_recursive_external_references_skip_checker_and_sidecars(self):
        model = tiny_model()
        model.graph.initializer.append(external_tensor("initializer"))
        nested = helper.make_graph([], "nested", [], [], [external_tensor("nested")])
        attribute_node = helper.make_node("Fixture", [], [], domain="tool.fixture")
        attribute_node.attribute.extend([
            helper.make_attribute("nested_graph", nested),
            helper.make_attribute("tensor", external_tensor("attribute")),
            helper.make_attribute("tensors", [external_tensor("attribute-list")]),
        ])
        sparse = helper.make_sparse_tensor(external_tensor("sparse-values"),
                                          external_tensor("sparse-indices"), [1])
        model.graph.sparse_initializer.append(sparse)
        attribute_node.attribute.append(helper.make_attribute("sparse", sparse))
        model.graph.node.append(attribute_node)
        function = helper.make_function("tool.fixture", "Local", ["x"], ["y"],
                                        [helper.make_node("Constant", [], ["y"],
                                                          value=external_tensor("function"))],
                                        [helper.make_opsetid("", 21)],
                                        attribute_protos=[helper.make_attribute(
                                            "default_tensor", external_tensor("function-default"))])
        model.functions.append(function)
        training = model.training_info.add()
        training.algorithm.CopyFrom(nested)
        digest, size = self.save(model)
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            self.assertEqual(path, self.model_path, "attempted to open a sidecar")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), patch(
            "builtins.open", side_effect=AssertionError("unexpected filesystem read")
        ), patch.object(onnx.numpy_helper, "to_array",
                        side_effect=AssertionError("tensor conversion must be skipped")), patch.object(
            onnx.checker, "check_model", side_effect=AssertionError("checker must be skipped")
        ):
            report = inspector.inspect_model(self.model_path, digest, size)
        names = [item["tensor_name"] for item in report["external_data"]]
        self.assertCountEqual(names, ["initializer", "nested", "attribute", "attribute-list",
                                     "sparse-values", "sparse-indices", "sparse-values",
                                     "sparse-indices", "function", "function-default", "nested"])
        self.assertEqual(report["local_functions"][0]["name"], "Local")
        self.assertEqual(report["verification"]["structural_checker"], "skipped_external_data")
        self.assertEqual(self.cli(digest, size).returncode, 2)
        self.assertEqual(json.loads(self.output.read_text())["status"], "blocked_external_data")
        self.assertFalse((self.root / "absent-sidecar.bin").exists())

    def test_success_replaces_report_and_same_path_is_rejected(self):
        digest, size = self.save(tiny_model())
        self.output.write_text("old report")
        self.assertEqual(self.cli(digest, size).returncode, 0)
        self.assertEqual(json.loads(self.output.read_text())["artifact"]["sha256"], digest)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["fixture.onnx", "report.json"])
        self.output = self.model_path
        self.assertNotEqual(self.cli(digest, size).returncode, 0)
        self.assertEqual(hashlib.sha256(self.model_path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
