# Intent Recognition

This module converts a natural-language command into a semantic action graph.
It combines an Abstract Meaning Representation (AMR) graph with token,
dependency, lemma, and named-entity information from spaCy.

> **Note:** The directory name is `intent_recogniton` (without the second
> `i` in “recognition”). Use that spelling in imports and paths.

## Processing pipeline

1. `PredicateGraphs` parses the sentence into an AMR graph.
2. `NestedIntents` converts the AMR graph into nested dictionaries and derives
   action frames.
3. `SpacyActionGraph` extracts linguistic information from the sentence.
4. `SemanticActionGraph` merges both representations into a normalized graph.

The resulting graph contains:

- `root`: ID of the root action.
- `nodes`: actions and their related entities, times, quantities, and units.
- `edges`: labeled relationships between nodes, such as `actor`, `recipient`,
  `time`, and `nested_action`.

## Installation

From this directory, install the Python dependencies:

```bash
python3 -m pip install -r req.txt
```

The AMR model is downloaded automatically the first time `PredicateGraphs` is
initialized. This requires an internet connection and may take some time.

## Usage

Run the example from the repository root:

```bash
python3 -m modules.intent_recogniton.main
```

Use the pipeline from Python:

```python
from modules.intent_recogniton.main import parse

parse("Remind John to call Mary tomorrow.")
```

The `parse` function prints the combined graph as formatted JSON.

For direct access to the structured result, compose the individual stages:

```python
from modules.intent_recogniton.nested_intents import NestedIntents
from modules.intent_recogniton.predicate_graphs import PredicateGraphs
from modules.intent_recogniton.semantic_action_graph import SemanticActionGraph
from modules.intent_recogniton.spacy_action_graph import SpacyActionGraph

sentence = "Remind John to call Mary tomorrow."

amr_graph = PredicateGraphs().sent_to_graph(sentence)
nested = NestedIntents().decompose(amr_graph)
actions = NestedIntents().derive_actions(nested)
tokens = SpacyActionGraph().analyze(sentence)

graph = SemanticActionGraph().combine(
    actions,
    tokens,
    root_action=nested["id"],
)
```

## Files

- `main.py`: assembles and runs the complete pipeline.
- `predicate_graphs.py`: creates PENMAN AMR graphs from sentences.
- `nested_intents.py`: decomposes AMR graphs and extracts action frames.
- `spacy_action_graph.py`: performs spaCy token and dependency analysis.
- `semantic_action_graph.py`: creates and formats the final semantic graph.
- `data.py`: contains sample voice-assistant commands.
- `req.txt`: lists the module dependencies.

## Current role mappings

`SemanticActionGraph` includes explicit mappings for `remind` and `call`.
Unknown AMR roles are retained without their leading colon. Additional mappings
can be supplied when constructing `SemanticActionGraph`:

```python
custom_roles = {
    "schedule": {
        ":ARG0": "actor",
        ":ARG1": "event",
        ":time": "time",
    }
}

semantic_graph = SemanticActionGraph(role_map=custom_roles)
```
