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

        for action in amr_actions:
            for argument in action["arguments"]:
                role = self.role_map.get(action["action"], {}).get(
                    argument["role"], argument["role"].removeprefix(":")
                )
                if "value" in argument or role == "requester":
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
                edges.append({
                    "source": action["id"],
                    "relation": (
                        "nested_action" if argument["type"] == "action" else role
                    ),
                    "target": argument["target"],
                })

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

        return {
            "root": root_action or (amr_actions[0]["id"] if amr_actions else None),
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
