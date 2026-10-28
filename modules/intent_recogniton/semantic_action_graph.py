import json


class SemanticActionGraph:
    """Merge normalized AMR actions and spaCy lemmas into a compact graph."""

    ROLE_MAP = {
        "remind": {
            ":ARG0": "requester",
            ":ARG1": "content",
            ":ARG2": "recipient",
        },
        "call": {
            ":ARG0": "actor",
            ":ARG1": "recipient",
            ":time": "time",
        },
    }

    def __init__(self, role_map: dict | None = None):
        self.role_map = {**self.ROLE_MAP, **(role_map or {})}

    def _ner_type(
        self,
        label: str,
        role: str,
        tokens: list[dict],
    ) -> str | None:
        labels = label.lower().split()
        matches = [
            token for token in tokens
            if token["text"].lower() in labels and token.get("ner_type")
        ]
        if matches and all(
            token["ner_type"] == matches[0]["ner_type"] for token in matches
        ):
            return matches[0]["ner_type"]
        if role in {"actor", "recipient"} and any(
            token["text"].lower() in labels and token["pos"] == "PROPN"
            for token in tokens
        ):
            return "PERSON"
        return None

    def _nested_node_type(self, role: str, predicate: str | None = None) -> str:
        role = role.removeprefix(":")
        if predicate == "mass-quantity":
            return "measurement"
        if role == "quant":
            return "quantity"
        if role == "unit":
            return "unit"
        if role in {"time", "year", "month", "day", "weekday"}:
            return "time"
        return "entity"

    def _add_nested_arguments(
        self,
        parent_id: str,
        arguments: dict,
        nodes: dict,
        edges: list[dict],
        expanded: set[str],
        edge_keys: set[tuple],
    ) -> None:
        if parent_id in expanded:
            return
        expanded.add(parent_id)

        for role, targets in arguments.items():
            if role == ":name":
                continue

            relation = role.removeprefix(":")
            for index, target in enumerate(targets):
                if isinstance(target, dict):
                    target_id = target.get("id") or target.get("reference")
                    if not target_id:
                        continue
                    predicate = target.get("predicate")
                    label = (
                        predicate.replace("-", " ")
                        if predicate
                        else target_id
                    )
                    nodes.setdefault(target_id, {
                        "id": target_id,
                        "type": self._nested_node_type(role, predicate),
                        "label": label,
                    })
                    child_arguments = target.get("arguments", {})
                else:
                    role_id = relation.replace("-", "_")
                    target_id = f"{parent_id}_{role_id}_{index}"
                    nodes.setdefault(target_id, {
                        "id": target_id,
                        "type": self._nested_node_type(role),
                        "label": str(target).strip('"'),
                    })
                    child_arguments = {}

                edge_key = (parent_id, relation, target_id)
                if edge_key not in edge_keys:
                    edges.append({
                        "source": parent_id,
                        "relation": relation,
                        "target": target_id,
                    })
                    edge_keys.add(edge_key)

                if child_arguments:
                    self._add_nested_arguments(
                        target_id,
                        child_arguments,
                        nodes,
                        edges,
                        expanded,
                        edge_keys,
                    )

    def combine(
        self,
        amr_actions: list[dict],
        spacy_tokens: list[dict],
        root_action: str | None = None,
    ) -> dict:
        spacy_lemmas = {
            token["lemma"].lower(): token["lemma"]
            for token in spacy_tokens
        }
        nodes = {
            action["id"]: {
                "id": action["id"],
                "type": "action",
                "label": spacy_lemmas.get(action["action"], action["action"]),
            }
            for action in amr_actions
        }
        edges = []
        edge_keys = set()
        expanded = set()

        for action in amr_actions:
            for index, argument in enumerate(action["arguments"]):
                role = self.role_map.get(action["action"], {}).get(
                    argument["role"], argument["role"].removeprefix(":")
                )
                if role == "requester":
                    continue

                if "value" in argument:
                    value_id = f"{action['id']}_{role}_{index}"
                    nodes[value_id] = {
                        "id": value_id,
                        "type": self._nested_node_type(argument["role"]),
                        "label": str(argument["value"]).strip('"'),
                    }
                    edges.append({
                        "source": action["id"],
                        "relation": role,
                        "target": value_id,
                    })
                    edge_keys.add((action["id"], role, value_id))
                    continue

                target_node = {
                    "id": argument["target"],
                    "type": "time" if role == "time" else argument["type"],
                    "label": argument["label"],
                }
                ner_type = self._ner_type(
                    argument["label"], role, spacy_tokens
                )
                if ner_type:
                    target_node["ner_type"] = ner_type
                nodes.setdefault(argument["target"], target_node)
                edge = {
                    "source": action["id"],
                    "relation": (
                        "nested_action" if argument["type"] == "action" else role
                    ),
                    "target": argument["target"],
                }
                edge_key = (edge["source"], edge["relation"], edge["target"])
                if edge_key not in edge_keys:
                    edges.append(edge)
                    edge_keys.add(edge_key)

                if argument["type"] != "action":
                    self._add_nested_arguments(
                        argument["target"],
                        argument.get("arguments", {}),
                        nodes,
                        edges,
                        expanded,
                        edge_keys,
                    )

        actor_targets = {
            edge["target"] for edge in edges
            if edge["relation"] == "actor"
        }
        edges = [
            edge for edge in edges
            if not (
                edge["source"] == root_action
                and edge["relation"] == "recipient"
                and edge["target"] in actor_targets
            )
        ]
        referenced = {edge["source"] for edge in edges} | {
            edge["target"] for edge in edges
        }

        root = root_action if root_action in nodes else (
            amr_actions[0]["id"] if amr_actions else None
        )
        return {
            "root": root,
            "nodes": [node for node_id, node in nodes.items() if node_id in referenced],
            "edges": edges,
        }

    def format(self, graph: dict) -> str:
        node_lines = ",\n".join(
            f"    {json.dumps(node)}" for node in graph["nodes"]
        )
        edge_lines = ",\n".join(
            f"    {json.dumps(edge)}" for edge in graph["edges"]
        )
        return (
            "{\n"
            f'  "root": {json.dumps(graph["root"])},\n'
            '  "nodes": [\n'
            f"{node_lines}\n"
            "  ],\n"
            '  "edges": [\n'
            f"{edge_lines}\n"
            "  ]\n"
            "}"
        )
