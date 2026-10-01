"""Duplicate-key-safe YAML loading shared by policy and registry files."""

import yaml  # type: ignore[import-untyped]
from yaml.constructor import ConstructorError  # type: ignore[import-untyped]
from yaml.nodes import MappingNode  # type: ignore[import-untyped]


class UniqueKeySafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML loader that rejects duplicate mapping keys."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def load_unique_keys_yaml(stream: object) -> object:
    """Load YAML content, rejecting duplicate mapping keys."""
    return yaml.load(stream, Loader=UniqueKeySafeLoader)
