
# Outlook:
"""
                Natural Language
                       │
                       ▼
                  AMR Parser
                   (Paper 1)
                       │
                       ▼
                 Semantic Graph
                       │
              ┌────────┴────────┐
              ▼                 ▼
       Graph Embedding     Graph Structure
              │                 │
              ▼                 │
       Intent Clustering        │
          (Paper 2)             │
              │                 │
              └────────┬────────┘
                       ▼
              Graph Generalization
                       │
                       ▼
              Intent Class + Slots
"""
from .predicate_graphs import PredicateGraphs
from .nested_intents import NestedIntents
from .spacy_action_graph import SpacyActionGraph
from .semantic_action_graph import SemanticActionGraph

def parse(sentence = "Remind John to call Mary tomorrow."):
   parser = PredicateGraphs()
   nested_intents = NestedIntents()
   spacy_action_graph = SpacyActionGraph()
   semantic_action_graph = SemanticActionGraph()

   graph = parser.sent_to_graph(sentence)

   decomposed_intents = nested_intents.decompose(graph)
   amr_actions = nested_intents.derive_actions(decomposed_intents)
   spacy_tokens = spacy_action_graph.analyze(sentence)
   combined_graph = semantic_action_graph.combine(
       amr_actions,
       spacy_tokens,
       root_action=decomposed_intents["id"],
   )
   print(semantic_action_graph.format(combined_graph))
   return semantic_action_graph

if __name__ == "__main__":
   parse()
