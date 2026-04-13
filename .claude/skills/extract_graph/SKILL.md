---
name: extract_graph
description: Guide the agent through the full iReC knowledge-graph pipeline — corpus preprocessing, SPaR.txt term extraction, acronym and Uniclass concept gathering, and SKOS+IREC graph construction. Invoke with an optional path to a new PDF or HTML corpus, e.g. `/extract_graph data/my_new_corpus.pdf`.
argument-hint: [optional: path to new PDF or HTML corpus file]
---

You are running the iReC knowledge-graph extraction pipeline. Work through each stage below in order. At every stage:

1. Check the preconditions.
2. Run the required notebook cells or commands.
3. Inspect the output and report a brief summary to the user.
4. **Stop and ask the user to confirm** before moving to the next stage.

Never skip a stage or batch stages together without explicit user approval.

## Arguments

`$ARGUMENTS` may contain a path to a new corpus file (PDF or HTML). If provided:
- Add the file as a new **foreground** corpus entry alongside the Merged Approved Documents.
- Treat the existing EU regulation HTML files as the background corpus unchanged.
- Name any new intermediate files after the source filename (stem), e.g. `my_new_corpus_processed.json`.

If `$ARGUMENTS` is empty, run the pipeline on the default corpus (Merged Approved Documents as foreground, EU regulation HTMLs as background).

---

## Precondition checks — run these first

Before starting any stage, verify:

```bash
# 1. Correct Python environment
python --version   # must be 3.9.x

# 2. SPaR.txt container is running
curl -s http://localhost:8501/ | head -c 100
# If this fails, start the container:
# docker start SPaR_API
# or build it if it doesn't exist yet:
# docker build --build-arg ONLY_CODE=$(date +%s) ./SPaR.txt/ -t spar
# docker run --name SPaR_API -p 8501:8501 spar

# 3. Required input files exist
ls data/term_extraction_input/
# Expected: The Merged Approved Documents.pdf, uniclass_2015.ttl, *.html (EU regulations)

# 4. IREC ontology schema exists
ls data/graph_data/IREC.rdf
```

Report the result of each check to the user. If SPaR.txt is not running, start it and wait for it to be healthy before proceeding.

---

## Stage 1 — Preprocessing and term extraction

**Notebook:** `1. Term Extraction.ipynb`, sections 1 and 2.

**What happens:**
- `read_pdf` extracts page-level text from the Merged Approved Documents PDF (and any new corpus PDF passed as `$ARGUMENTS`).
- `grab_HTML_text_simple` + `convert_html_to_mydoc` extract paragraph-level text from the EU regulation HTML files.
- Each document is wrapped in a `CustomDocument` and saved as a JSON file alongside the source.
- `NER.process_custom_document` sends texts in batches of 80 to the SPaR.txt API (`http://localhost:8501/predict_objects/`) and writes `SPaR_labels` back into the JSON. Processing resumes from the last completed batch if interrupted.

**Run the cells:**
```bash
jupyter nbconvert --to notebook --execute "1. Term Extraction.ipynb" \
  --ExecutePreprocessor.timeout=3600 \
  --output "1. Term Extraction.ipynb"
```

Note: SPaR.txt extraction takes ~20 minutes on CPU. If a new corpus was provided in `$ARGUMENTS`, ensure it is wired into the notebook's corpus list before executing — read the import cell at the top and add the new path there using Edit.

**Inspect the output:**
```python
import pickle, json, glob
# Count sentences and extracted spans per document
for fp in glob.glob("data/term_extraction_input/*.json"):
    with open(fp) as f:
        doc = json.load(f)
    spans = [c['meta']['SPaR_labels'] for c in doc['content'] if c['meta'].get('SPaR_labels')]
    print(fp, "—", len(spans), "sections with spans")
```

Report the sentence and span counts. Flag any document with zero spans as a likely preprocessing failure.

**Pause:** Ask the user to confirm the extraction counts look reasonable before continuing.

---

## Stage 2 — Filtering domain-specific terms

**Notebook:** `1. Term Extraction.ipynb`, section 3.

**What happens:**
1. All SPaR.txt object spans are collected from foreground and background corpus JSONs.
2. `RegexFilter.run_filter` removes spans matching noise patterns.
3. `custom_cleaning_rules`, `remove_determiners` apply text-level cleaning.
4. Embeddings are computed for all unique spans (IDF-weighted using `IDF_computation`).
5. Domain specificity is measured by comparing foreground and background nearest-neighbour distributions + a KL-divergence-inspired score.
6. Spans above the domain-specificity threshold are saved as `data/graph_data/domain_terms.pkl`.

**Run the cells** (section 3 only — use cell tags or run the full notebook again if sections cannot be isolated):
```bash
jupyter nbconvert --to notebook --execute "1. Term Extraction.ipynb" \
  --ExecutePreprocessor.timeout=3600 \
  --output "1. Term Extraction.ipynb"
```

**Inspect the output:**
```python
import pickle
domain_terms = pickle.load(open("data/graph_data/domain_terms.pkl", "rb"))
print(f"Domain terms: {len(domain_terms)}")
print("Sample:", domain_terms[:20])
```

Report the count and a random sample of 20 terms. Flag if the count is 0 or unexpectedly low (< 100 would be suspicious).

**Pause:** Ask the user to confirm the domain term list looks sensible.

---

## Stage 3 — Acronym extraction

**Notebook:** `2. Grab additional terms.ipynb`, section 1 ("Grab acronyms from text").

**What happens:**
- For each span in `domain_terms`, the notebook scans foreground corpus sentences for patterns like `span (ABBR)`.
- `is_acronym` validates that the abbreviation initials match the span words.
- Results saved to `data/graph_data/acronyms_found_in_text.pkl` as a `{acronym: [full_span, ...]}` dict.

**Run the cells:**
```bash
jupyter nbconvert --to notebook --execute "2. Grab additional terms.ipynb" \
  --ExecutePreprocessor.timeout=600 \
  --output "2. Grab additional terms.ipynb"
```

**Inspect the output:**
```python
import pickle
acronyms = pickle.load(open("data/graph_data/acronyms_found_in_text.pkl", "rb"))
print(f"Acronyms found: {len(acronyms)}")
print("Sample:", dict(list(acronyms.items())[:10]))
```

**Pause:** Confirm with the user before continuing.

---

## Stage 4 — Uniclass term matching

**Notebook:** `2. Grab additional terms.ipynb`, sections 2 and 3 ("Prepare Uniclass terms").

**What happens:**
- `group_ttl_lines` + `grab_uids_and_labels_with_definition` parse `data/term_extraction_input/uniclass_2015.ttl` into a `{uid: {pref_label, alt_labels, definition}}` dict.
- `grab_nodes` caches this as `uniclass_2015.ttl.json` so parsing only happens once.
- Each Uniclass `pref_label` is searched (case-insensitive) in foreground corpus paragraphs.
- Matching terms saved to `data/graph_data/uniclass_terms_in_text.pkl`.

**Run the cells** (already covered by the Stage 3 `nbconvert` call above if run in sequence; otherwise re-run notebook 2).

**Inspect the output:**
```python
import pickle
uniclass_terms = pickle.load(open("data/graph_data/uniclass_terms_in_text.pkl", "rb"))
total_uniclass = len(pickle.load(open("data/term_extraction_input/uniclass_2015.ttl.json"
    if __import__('pathlib').Path("data/term_extraction_input/uniclass_2015.ttl.json").exists()
    else "uniclass_2015.ttl.json", "rb")) if False else {})
print(f"Uniclass terms matched in corpus: {len(uniclass_terms)}")
print("Sample:", [v['pref_label'] for v in list(uniclass_terms.values())[:10]])
```

Or more simply:
```python
import pickle
uniclass_terms = pickle.load(open("data/graph_data/uniclass_terms_in_text.pkl", "rb"))
print(f"Uniclass terms matched: {len(uniclass_terms)}")
print("Sample:", [v['pref_label'] for v in list(uniclass_terms.values())[:10]])
```

**Pause:** Confirm with the user before building the graph.

---

## Stage 5 — Graph initialisation

**Notebook:** `3. Create graph.ipynb`, section "Prepare graph".

**What happens:**
- `rdflib.Graph` is created and `data/graph_data/IREC.rdf` is parsed into it (this loads the IREC ontology schema, which defines `IREC.span` and other class/property terms).
- Namespaces are bound:

  | Prefix | URI |
  |--------|-----|
  | `spans` | `https://spans.irec.org/#` |
  | `concepts` | `https://concepts.irec.org/#` |
  | `uniclass` | `https://www.example.org/uniclass/#` |
  | `wiki` | `https://www.wikidata.org/wiki/` |
  | `skos` | `http://www.w3.org/2004/02/skos/core#` |
  | `prov` | `http://www.w3.org/ns/prov#` |
  | `dct` | `http://purl.org/dc/terms/#` |

- Two `skos:ConceptScheme` nodes are registered via `add_scheme_uid`: one for SPANS, one for CONCEPTS.
- Primary source IRIs are declared (`merged_approved_documents_IRI`, `uniclass_IRI`, `spart_txt_IRI`).

**Key graph-building conventions — always follow these:**
- **Adding nodes**: use `UID_assigner.assign_UID(text, namespace)` to get a deterministic UID, then call the appropriate wrapper (`irec_span`, `skos_node`, etc.) followed by `add_tuples(graph, triples)`.
- **Never call `graph.add()` directly** — always go through `add_tuples`, which deduplicates.
- **Every node gets provenance**: call `provenance(uid, source_IRI, namespace)` and `prov_agent(uid, agent_IRI, namespace)`.

**Run the initialisation cells only** (up to and including the `add_scheme_uid` calls):
```bash
# Run notebook 3 up to cell N — if you cannot isolate cells,
# run the full notebook but check the intermediate TTL saved after this stage.
jupyter nbconvert --to notebook --execute "3. Create graph.ipynb" \
  --ExecutePreprocessor.timeout=1800 \
  --output "3. Create graph.ipynb"
```

**Inspect:**
```python
from rdflib import Graph
g = Graph()
g.parse("data/graph_data/approved_doc_terms_only.ttl", format="turtle")
print(f"Triples after init: {len(g)}")
```

**Pause:** Confirm the graph loaded without errors.

---

## Stage 6 — Add SPANS and CONCEPTS from the Approved Documents

**Notebook:** `3. Create graph.ipynb`, sections "Add domain terms", "Add Acronyms", "Add CONCEPTS", "Add SPANS (glossary/index terms)".

**What happens for each node type:**

| Node type | Namespace | SKOS/IREC triples added |
|-----------|-----------|------------------------|
| SPaR.txt domain span | `SPANS` | `IREC.span` type, `skos:inScheme`, `prov:hadPrimarySource` (Approved Docs), `prov:wasAttributedTo` (SPaR.txt) |
| Acronym span | `SPANS` | same as above; additionally linked to its full-form span via `skos:exactMatch` |
| Defined concept | `CONCEPTS` | `skos:Concept` type, `skos:inScheme`, `skos:definition`, `dct:title`, `prov:hadPrimarySource`, optionally `skos:altLabel`, `skos:broader`, `skos:related` |
| Index/glossary span | `SPANS` | `IREC.span` type, `skos:inScheme`, optionally `skos:broader`, `skos:related`; no provenance if manually curated |

Spans and concepts are linked via `skos:exactMatch` wherever a span string matches a concept label.

**Run:** covered by the full notebook 3 execution above. The intermediate checkpoint file `approved_doc_terms_only.ttl` is written after this section — verify it exists.

**Inspect:**
```python
from rdflib import Graph, Namespace, SKOS, RDF
g = Graph()
g.parse("data/graph_data/approved_doc_terms_only.ttl", format="turtle")
SPANS = Namespace("https://spans.irec.org/#")
CONCEPTS = Namespace("https://concepts.irec.org/#")
n_spans = sum(1 for _ in g.subjects(RDF.type, None) if str(next(iter(g.subjects()), '')).startswith(str(SPANS)))
print(f"Total triples: {len(g)}")
# Simpler count:
print(f"Span nodes: {sum(1 for s in g.subjects() if 'spans.irec.org' in str(s))}")
print(f"Concept nodes: {sum(1 for s in g.subjects() if 'concepts.irec.org' in str(s))}")
```

**Pause:** Confirm counts look right before adding Uniclass.

---

## Stage 7 — Add Uniclass concepts

**Notebook:** `3. Create graph.ipynb`, section "Add Uniclass terms".

**What happens:**
- Each matched Uniclass entry gets a node in the `UNICLASS` namespace, with `skos:prefLabel`, optional `skos:altLabel`, optional `skos:definition`, and `prov:hadPrimarySource` pointing to `uniclass_IRI`.
- Each Uniclass concept is linked to any matching SPANS node via `skos:exactMatch`.
- `skos:broader` edges within Uniclass are preserved if the parent term also appears in the corpus.

**Inspect after running:**
```python
from rdflib import Graph
g = Graph()
g.parse("data/graph_data/initial_graph.ttl", format="turtle")
print(f"Total triples (with Uniclass): {len(g)}")
print(f"Uniclass nodes: {sum(1 for s in g.subjects() if 'example.org/uniclass' in str(s))}")
```

**Pause:** Confirm before serialisation.

---

## Stage 8 — Serialise final graph

**Notebook:** `3. Create graph.ipynb`, final serialisation cell.

Serialise to both formats:

```python
from pathlib import Path
graph_output_fp = Path("data/graph_data")

# Turtle — for GraphDB import and human inspection
irec_graph.serialize(destination=str(graph_output_fp / "initial_graph.ttl"), format="turtle")

# RDF/XML — for programmatic rdflib loading in downstream GraphRAG code
irec_graph.serialize(destination=str(graph_output_fp / "IREC_updated.rdf"), format="xml")

print(f"Final graph: {len(irec_graph)} triples")
```

If a new corpus was added (`$ARGUMENTS` was non-empty), name the output files with a suffix, e.g. `initial_graph_with_<stem>.ttl`, to avoid overwriting the baseline graph.

**Final verification:**
```bash
# File sizes — a healthy graph is typically several MB
ls -lh data/graph_data/initial_graph.ttl data/graph_data/IREC_updated.rdf

# Quick SPARQL sanity check
python3 - <<'EOF'
from rdflib import Graph, SKOS
g = Graph()
g.parse("data/graph_data/initial_graph.ttl", format="turtle")
results = list(g.query("SELECT (COUNT(?s) AS ?n) WHERE { ?s a <http://www.w3.org/2004/02/skos/core#Concept> }"))
print("skos:Concept count:", results[0][0])
results2 = list(g.query("SELECT (COUNT(?s) AS ?n) WHERE { ?s ?p ?o . FILTER(CONTAINS(STR(?s), 'spans.irec.org')) }"))
print("SPANS triples:", results2[0][0])
EOF
```

Report the final triple count, concept count, and file sizes to the user.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ConnectionRefusedError` on SPaR.txt call | Container not running | `docker start SPaR_API` |
| Zero domain terms after filtering | Embedding threshold too aggressive, or foreground JSON is empty | Check that `SPaR_labels` fields are populated in the corpus JSONs |
| `KeyError` in `UID_assigner` | UID for a namespace not initialised | Call `add_scheme_uid` for that namespace before calling `assign_UID` |
| Duplicate triples warning | `add_tuples` deduplicates silently — not an error | Safe to ignore |
| Notebook execution timeout | SPaR.txt is slow on CPU | Increase `--ExecutePreprocessor.timeout` or run cells manually |
| New corpus PDF has no text extracted | PDF is image-based / scanned | `pdftotext` (installed via conda) requires selectable text; OCR is out of scope |
