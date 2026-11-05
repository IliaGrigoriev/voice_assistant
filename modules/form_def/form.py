import yaml


class Form:
    TYPES = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
    }

    def __init__(self, path: str):
        with open(path) as f:
            self.definition = yaml.safe_load(f)

        self.nodes = {
            node["id"]: node
            for node in self.definition["nodes"]
        }

        self.values = {
            node_id: {}
            for node_id in self.nodes
        }

    def fill(self, node_id: str, **kwargs):
        """Fill slots belonging to an action."""

        node = self.nodes[node_id]

        for name, value in kwargs.items():
            slot = node.get("slots", {}).get(name)

            if slot is None:
                raise ValueError(f"Unknown slot: {name}")

            expected_type = self.TYPES[slot["type"]]

            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{name} must be {expected_type.__name__}"
                )

            self.values[node_id][name] = value

    def missing(self):
        """Return all missing required slots."""

        missing = {}

        for node_id, node in self.nodes.items():
            required = [
                name
                for name, slot in node.get("slots", {}).items()
                if slot.get("required", False)
                and name not in self.values[node_id]
            ]

            if required:
                missing[node_id] = required

        return missing

    def is_complete(self):
        return not self.missing()

    def get(self, node_id: str):
        return self.values[node_id].copy()