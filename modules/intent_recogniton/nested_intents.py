import re

import penman


class NestedIntents:
    _ACTION = re.compile(r"(.+)-(\d+)$")

    # Decompose a PENMAN graph into nested intents
    # ---------------------------------------------------------------------
    def decompose(self, graph: penman.Graph) -> dict:
        concepts = {
            instance.source: instance.target
            for instance in graph.instances()
        }

        adjacency = {}
        for edge in graph.edges():
            adjacency.setdefault(edge.source, []).append(
                (edge.role, edge.target)
            )

        attributes = {}
        for attribute in graph.attributes():
            attributes.setdefault(attribute.source, []).append(
                (attribute.role, attribute.target)
            )

        def visit(node, visited):
            if node in visited:
                return {"reference": node}

            visited = visited | {node}
            arguments = {}

            for role, target in adjacency.get(node, []):
                arguments.setdefault(role, []).append(
                    visit(target, visited)
                )

            for role, value in attributes.get(node, []):
                arguments.setdefault(role, []).append(value)

            return {
                "id": node,
                "predicate": concepts[node],
                "arguments": arguments,
            }

        return visit(graph.top, set())

    # Derive framed actions from the nested AMR structure
    # ---------------------------------------------------------------------
    def derive_actions(self, decomposed: dict) -> list[dict]:
        nodes = {}
        stack = [decomposed]
        while stack:
            node = stack.pop()
            if "predicate" not in node:
                continue
            nodes.setdefault(node["id"], node)
            for children in node.get("arguments", {}).values():
                stack.extend(
                    child for child in children
                    if isinstance(child, dict) and "predicate" in child
                )

        action_ids = {
            node_id for node_id, node in nodes.items()
            if self._ACTION.fullmatch(node["predicate"])
        }
        actions = []
        for node_id, node in nodes.items():
            match = self._ACTION.fullmatch(node["predicate"])
            if not match:
                continue

            arguments = []
            for role, targets in node["arguments"].items():
                for target in targets:
                    if not isinstance(target, dict):
                        arguments.append({"role": role, "value": target})
                        continue

                    target_id = target.get("id") or target.get("reference")
                    target = nodes.get(target_id, target)
                    name = target.get("arguments", {}).get(":name", [])
                    if name:
                        parts = [
                            str(value).strip('"')
                            for key, values in name[0]["arguments"].items()
                            if key.startswith(":op")
                            for value in values
                        ]
                        label = " ".join(parts)
                    else:
                        label = re.sub(
                            r"-\d+$", "", target.get("predicate", target_id)
                        ).replace("-", " ")

                    arguments.append({
                        "role": role,
                        "target": target_id,
                        "label": label,
                        "type": "action" if target_id in action_ids else "entity",
                        "arguments": target.get("arguments", {}),
                    })

            actions.append({
                "id": node_id,
                "action": match.group(1).replace("-", " "),
                "predicate": node["predicate"],
                "arguments": arguments,
            })

        return actions
