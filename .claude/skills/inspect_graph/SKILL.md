---
name: inspect_graph
description: Inspect the iReC knowledge graph (initial_graph.ttl). Use this before working on GraphRAG retrieval logic to understand which predicates, node types, and relations are actually present in the graph, and to sample specific nodes by type or label.
argument-hint: [optional: node label or URI fragment to inspect, e.g. "door"]
---

Run the following inspection steps using `poetry run python -c "..."`. Adapt as needed based on `$ARGUMENTS`.

## Step 1 — Predicate inventory

Always start here to see what relations are actually in the graph and how frequently they appear:

```python
from collections import Counter
from rdflib import Graph

g = Graph()
g.parse("data/graph_data/initial_graph.ttl", format="turtle")
print(f"Total triples: {len(g)}\n")

pred_counts = Counter(
    str(p).split("#")[-1].split("/")[-1]
    for _, p, _ in g
)
for pred, count in pred_counts.most_common(30):
    print(f"  {count:>6}  {pred}")
```

## Step 2 — Node type counts

```python
from rdflib import Graph, Namespace
from rdflib.namespace import RDF, SKOS

g = Graph()
g.parse("data/graph_data/initial_graph.ttl", format="turtle")

IREC = Namespace("https://schema.irec.org/#")

spans    = sum(1 for _ in g.subjects(RDF.type, IREC.CharacterSpan))
concepts = sum(1 for _ in g.subjects(RDF.type, SKOS.Concept))
print(f"CharacterSpan (SPANS)   : {spans}")
print(f"skos:Concept (CONCEPTS) : {concepts}")
print(f"skos:exactMatch triples : {sum(1 for _ in g.triples((None, SKOS.exactMatch, None)))}")
print(f"skos:definition triples : {sum(1 for _ in g.triples((None, SKOS.definition, None)))}")
print(f"skos:broader triples    : {sum(1 for _ in g.triples((None, SKOS.broader, None)))}")
print(f"skos:related triples    : {sum(1 for _ in g.triples((None, SKOS.related, None)))}")
```

## Step 3 — Sample a CONCEPT node

Replace `"ventilation"` with the label you want to inspect (or use `$ARGUMENTS`):

```python
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, SKOS

g = Graph()
g.parse("data/graph_data/initial_graph.ttl", format="turtle")

IREC     = Namespace("https://schema.irec.org/#")
CONCEPTS = Namespace("https://concepts.irec.org/#")

label = "ventilation"   # <-- change this or use $ARGUMENTS

# find node by prefLabel
matches = [s for s, p, o in g.triples((None, SKOS.prefLabel, None))
           if str(o).lower() == label.lower()]

for node in matches:
    print(f"Node: {node}")
    for s, p, o in g.triples((node, None, None)):
        pred = str(p).split("#")[-1].split("/")[-1]
        print(f"  {pred:<30} {str(o)[:100]}")
```

## Step 4 — Sample a SPAN node

```python
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS

g = Graph()
g.parse("data/graph_data/initial_graph.ttl", format="turtle")

IREC  = Namespace("https://schema.irec.org/#")
SPANS = Namespace("https://spans.irec.org/#")

label = "ventilation"   # <-- change this

matches = [s for s, p, o in g.triples((None, RDFS.label, None))
           if str(o).lower() == label.lower()
           and str(s).startswith(str(SPANS))]

for node in matches:
    print(f"Node: {node}")
    for s, p, o in g.triples((node, None, None)):
        pred = str(p).split("#")[-1].split("/")[-1]
        print(f"  {pred:<30} {str(o)[:100]}")
    # also check what points TO this node
    print("  -- incoming --")
    for s, p, o in g.triples((None, None, node)):
        pred = str(p).split("#")[-1].split("/")[-1]
        src  = str(s).split("#")[-1].split("/")[-1]
        print(f"  {src:<30} --[{pred}]-->")
```

## Step 5 — Trace a concept's full neighbourhood (1 hop)

Useful for validating what a GraphRAG retriever would return for a given seed concept:

```python
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import SKOS, RDFS

g = Graph()
g.parse("data/graph_data/initial_graph.ttl", format="turtle")

seed_label = "ventilation"   # <-- change this

# find seed nodes
seeds = {s for s, p, o in g.triples((None, SKOS.prefLabel, None))
         if str(o).lower() == seed_label.lower()}
seeds |= {s for s, p, o in g.triples((None, RDFS.label, None))
          if str(o).lower() == seed_label.lower()}

triples = []
for seed in seeds:
    triples += list(g.triples((seed, None, None)))
    triples += list(g.triples((None, None, seed)))

print(f"1-hop neighbourhood of '{seed_label}': {len(triples)} triples")
for s, p, o in triples:
    s_label = str(s).split("#")[-1].split("/")[-1][:40]
    p_label = str(p).split("#")[-1].split("/")[-1]
    o_label = str(o).split("#")[-1].split("/")[-1][:60] if isinstance(o, URIRef) else str(o)[:60]
    print(f"  {s_label:<40} --[{p_label}]--> {o_label}")
```

## Notes

- All scripts must be run with `poetry run python -c "..."` from `/Users/rubenk/dev/irec/`
- The graph is 2MB / ~38k triples — loading takes ~1s; don't reload it repeatedly in a loop
- SPANS URIs use URL-encoded text as the fragment (e.g. `spans.irec.org/#ventilation`)
- CONCEPTS URIs use integer IDs as the fragment (e.g. `concepts.irec.org/#42`)
- UNICLASS URIs use the Uniclass classification code (e.g. `uniclass/#EF_65_40`)
