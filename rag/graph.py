
# Knowledge graph: entity/relation extraction from chunks
import os
import re
import json
import random
import argparse
from typing import Literal
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from config import MODEL_NAME, MAX_WORKERS, VECTORSTORE_DIR
from pipeline import load_vectorstore

# stored next to the vectorstore: the graph refers to its chunk ids,
# so both must be rebuilt together
GRAPH_PATH       = os.path.join(VECTORSTORE_DIR, "graph.json")
EXTRACTIONS_PATH = os.path.join(VECTORSTORE_DIR, "extractions.json")
SYNONYMS_PATH    = os.path.join(VECTORSTORE_DIR, "synonyms.json")
OVERRIDES_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synonym_overrides.json")

EMBEDDING_MODEL  = "text-embedding-3-small"


# extraction schema

EntityType = Literal[
    "ecosystem",   # coral reef, mangrove forest, seagrass meadow
    "organism",    # pulsing soft coral, Rhizophora
    "phenomenon",  # storm surge, wave breaking, turbulence, scour
    "quantity",    # turbulent kinetic energy, Reynolds stress, wave height
    "method",      # ROMS, SPH, Swin Transformer, flume experiment
    "structure",   # artificial reef, breakwater, monopile
    "location",    # Bangladesh, Adriatic Sea, Kyushu
]

class Entity(BaseModel):
    name: str        = Field(description="Canonical name, lowercase, singular (e.g. 'artificial reef', 'roms')")
    type: EntityType
    description: str = Field(description="One sentence on what this chunk says about the entity")

class Relation(BaseModel):
    source: str      = Field(description="Name of an entity from the entities list")
    relation: str    = Field(description="Short verb phrase, e.g. 'dissipates', 'is simulated by', 'increases'")
    target: str      = Field(description="Name of an entity from the entities list")

class ChunkGraph(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


EXTRACTION_PROMPT = """You build a knowledge graph about coastal hydrodynamics and hazards
(waves, currents, storm surge, reefs, coastal vegetation, and how they are modeled).

From the excerpt below, extract:
- entities: only concepts a reader might ask a question about (max 8).
  Skip lab apparatus, numerical schemes, statistical tools, parameter values,
  authors, figures, institutions and generic words like "study", "results", "model".
  Only extract entities that are mentioned in the excerpt.
- relations: links between those entities that the excerpt explicitly states (max 8).
  Both source and target MUST be names from your entities list.
  One relation per pair of entities. Fewer is fine, zero is fine:
  never invent a relation just to connect entities.

Entity types:
- ecosystem: a natural habitat (coral reef, mangrove forest, seagrass meadow)
- organism: a species or group of living things
- phenomenon: a physical process or event (storm surge, wave attenuation, turbulence, scour)
- quantity: a measurable property (wave height, drag coefficient, Reynolds number)
- method: a model, simulation technique or AI approach (ROMS, SPH, graph neural network)
- structure: a man-made structure (artificial reef, breakwater, monopile)
- location: a named geographic place (Bangladesh, Adriatic Sea)
The examples in this list are illustrations, not content: do not extract them
unless they appear in the excerpt.

Use generic canonical names, lowercase and singular, so the same concept gets the same
name across papers (e.g. "wave attenuation" not "the attenuation of waves observed in our flume").

EXCERPT:
{chunk}"""


# LLM extraction

def extract_from_chunk(chunk_text: str) -> ChunkGraph:
    client = OpenAI()
    resp = client.chat.completions.parse(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(chunk=chunk_text)}],
        response_format=ChunkGraph,
        temperature=0,
    )
    return clean_graph(resp.choices[0].message.parsed, chunk_text)


def mentioned_in(name: str, text: str) -> bool:
    """loose check: at least one significant word of the name appears in the text"""
    words = [w for w in re.split(r"[^a-z0-9]+", name) if len(w) >= 4]
    return not words or any(w in text for w in words)


def clean_graph(graph: ChunkGraph, chunk_text: str) -> ChunkGraph:
    """lowercase names, drop entities not in the chunk, drop relations to unknown entities,
    one relation per pair"""
    text = chunk_text.lower()
    for e in graph.entities:
        e.name = e.name.strip().lower()
    graph.entities = [e for e in graph.entities if mentioned_in(e.name, text)]
    names = {e.name for e in graph.entities}

    relations, seen = [], set()
    for r in graph.relations:
        r.source, r.target = r.source.strip().lower(), r.target.strip().lower()
        pair = frozenset((r.source, r.target))
        if r.source in names and r.target in names and r.source != r.target and pair not in seen:
            seen.add(pair)
            relations.append(r)
    graph.relations = relations
    return graph


def extract_all(chunks: dict[str, str]) -> dict[str, ChunkGraph]:
    """chunks: {chunk_id: text} -> {chunk_id: ChunkGraph}, in parallel.
    Failed chunks are skipped (and retried on the next run thanks to the cache)."""
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_from_chunk, text): cid for cid, text in chunks.items()}
        for i, future in enumerate(as_completed(futures), 1):
            cid = futures[future]
            try:
                results[cid] = future.result()
                print(f"✓ {i}/{len(chunks)}")
            except Exception as e:
                print(f"✗ {i}/{len(chunks)} {cid}: {e}")
    return results


# extraction cache (so the graph can be rebuilt without new LLM calls)

def load_extractions() -> dict[str, ChunkGraph]:
    if not os.path.exists(EXTRACTIONS_PATH):
        return {}
    with open(EXTRACTIONS_PATH) as f:
        return {cid: ChunkGraph.model_validate(g) for cid, g in json.load(f).items()}


def save_extractions(extractions: dict[str, ChunkGraph]):
    with open(EXTRACTIONS_PATH, "w") as f:
        json.dump({cid: g.model_dump() for cid, g in extractions.items()}, f, indent=1)


# graph building

def paper_of(chunk_text: str) -> str:
    return chunk_text.split("\n", 1)[0].removeprefix("Paper: ").strip()


def name_key(name: str) -> str:
    """same key for spelling variants: 'Free-stream velocities' ~ 'freestream velocity'"""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    if key.endswith("s") and not key.endswith("ss"):
        key = key[:-1]
    return key


def canonical_names(extractions: dict[str, ChunkGraph]) -> dict[str, str]:
    """name -> canonical name (the most frequent spelling among its variants)"""
    variants = defaultdict(Counter)
    for g in extractions.values():
        for e in g.entities:
            variants[name_key(e.name)][e.name] += 1
    return {
        name: counts.most_common(1)[0][0]
        for counts in variants.values() for name in counts
    }


def build_graph(
    extractions: dict[str, ChunkGraph],
    chunks: dict[str, str],
    synonyms: dict[str, str] | None = None,
) -> nx.DiGraph:
    """one node per entity (spelling variants and synonyms merged), one edge per (source, target) pair.
    Nodes keep the chunks/papers mentioning them; edges keep every relation phrase."""
    G = nx.DiGraph()
    types = defaultdict(Counter)
    spelling = canonical_names(extractions)
    synonyms = synonyms or {}
    canon = {name: synonyms.get(c, c) for name, c in spelling.items()}

    for cid, g in extractions.items():
        paper = paper_of(chunks[cid])
        for e in g.entities:
            name = canon[e.name]
            if name not in G:
                G.add_node(name, aliases=[], chunks=[], papers=[], descriptions=[])
            node = G.nodes[name]
            if e.name != name and e.name not in node["aliases"]:
                node["aliases"].append(e.name)
            if cid not in node["chunks"]:
                node["chunks"].append(cid)
            if paper not in node["papers"]:
                node["papers"].append(paper)
            node["descriptions"].append(e.description)
            types[name][e.type] += 1
        for r in g.relations:
            source, target = canon[r.source], canon[r.target]
            if source == target:
                continue
            if not G.has_edge(source, target):
                G.add_edge(source, target, relations=[])
            G.edges[source, target]["relations"].append({"relation": r.relation, "chunk": cid})

    # when chunks disagree on the type, the most common one wins
    for name, counts in types.items():
        G.nodes[name]["type"] = counts.most_common(1)[0][0]
    return G


# synonym merging: embeddings find candidate pairs, the LLM confirms them

class PairVerdict(BaseModel):
    id: int
    name_a: str = Field(description="First name of the pair, copied exactly")
    name_b: str = Field(description="Second name of the pair, copied exactly")
    same: bool

class PairVerdicts(BaseModel):
    verdicts: list[PairVerdict]


SYNONYM_PROMPT = """For each numbered pair of entities from a coastal hydrodynamics knowledge graph,
decide if both names refer to the SAME concept (same=true) or not (same=false).

same=true: only wording, abbreviation, spelling or grammar differs
  e.g. "kc number" / "keulegan-carpenter number", "turbulence kinetic energy" / "turbulent kinetic energy"
same=false: related but distinct concepts, including
  - opposites or different directions/dimensions: "gauged" / "ungauged", "2d" / "3d", "horizontal" / "vertical"
  - a general concept vs a more specific one: "reynolds number" / "roughness reynolds number"
  - a quantity vs its rate, coefficient or model: "wave height" / "wave height model"
  - different named places, events, experiments or cases
When unsure, answer false.

PAIRS:
{pairs}"""


def embed_texts(texts: list[str]) -> np.ndarray:
    client = OpenAI()
    vecs = []
    for i in range(0, len(texts), 500):
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[i:i + 500])
        vecs += [d.embedding for d in resp.data]
    V = np.array(vecs)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def synonym_candidates(G: nx.DiGraph, k: int = 5, min_sim: float = 0.5) -> list[tuple[str, str]]:
    """each entity's k nearest neighbors, embedded as 'name: description'
    (names alone embed poorly: 'kc number' is far from 'keulegan-carpenter number')"""
    names = sorted(G.nodes)
    V = embed_texts([f"{n}: {G.nodes[n]['descriptions'][0]}" for n in names])
    S = V @ V.T
    np.fill_diagonal(S, 0)
    pairs = set()
    for i in range(len(names)):
        for j in np.argsort(-S[i])[:k]:
            if S[i, j] >= min_sim:
                pairs.add(tuple(sorted((names[i], names[j]))))
    return sorted(pairs)


def describe(G: nx.DiGraph, name: str) -> str:
    return f'"{name}" ({G.nodes[name]["type"]}: {G.nodes[name]["descriptions"][0]})'


def confirm_batch(G: nx.DiGraph, batch: list[tuple[str, str]]) -> list[tuple[str, str]]:
    lines = "\n".join(f"{i}. {describe(G, a)}  vs  {describe(G, b)}" for i, (a, b) in enumerate(batch))
    resp = OpenAI().chat.completions.parse(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": SYNONYM_PROMPT.format(pairs=lines)}],
        response_format=PairVerdicts,
        temperature=0,
    )
    # keep a verdict only if the names it repeats match the pair it claims to answer
    return [
        batch[v.id] for v in resp.choices[0].message.parsed.verdicts
        if v.same and 0 <= v.id < len(batch) and {v.name_a, v.name_b} == set(batch[v.id])
    ]


def confirm_synonyms(G: nx.DiGraph, pairs: list[tuple[str, str]], batch_size: int = 10) -> list[tuple[str, str]]:
    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    confirmed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(confirm_batch, G, b) for b in batches]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                confirmed += future.result()
                print(f"✓ {i}/{len(batches)}")
            except Exception as e:
                print(f"✗ {i}/{len(batches)}: {e}")
    return confirmed


def synonym_map(G: nx.DiGraph, confirmed: list[tuple[str, str]]) -> dict[str, str]:
    """group confirmed pairs without chaining: a group is merged only if every pair
    in it was confirmed (a clique), so A=B and B=C never merges an unconfirmed A=C.
    Every member maps to the group's most mentioned name."""
    C = nx.Graph(confirmed)
    mapping, used = {}, set()
    # largest cliques first; each name joins at most one group
    for clique in sorted(nx.find_cliques(C), key=len, reverse=True):
        group = [n for n in clique if n not in used]
        if len(group) < 2:
            continue
        used.update(group)
        canonical = max(group, key=lambda n: len(G.nodes[n]["chunks"]))
        mapping.update({n: canonical for n in group if n != canonical})
    return mapping


def load_synonyms() -> dict[str, str]:
    if not os.path.exists(SYNONYMS_PATH):
        return {}
    with open(SYNONYMS_PATH) as f:
        return json.load(f)


def save_synonyms(mapping: dict[str, str]):
    with open(SYNONYMS_PATH, "w") as f:
        json.dump(mapping, f, indent=1, sort_keys=True)


def apply_overrides(mapping: dict[str, str]) -> dict[str, str]:
    """manual review on top of the LLM merges (synonym_overrides.json, tracked in git):
    'remove' drops wrong merges, 'add' forces missed ones"""
    if not os.path.exists(OVERRIDES_PATH):
        return mapping
    with open(OVERRIDES_PATH) as f:
        overrides = json.load(f)
    mapping = {a: c for a, c in mapping.items() if a not in overrides.get("remove", [])}
    mapping.update(overrides.get("add", {}))

    # follow chains: 'time-averaged tke' -> 'tke' -> 'turbulent kinetic energy'
    def resolve(name, seen=()):
        target = mapping.get(name)
        return name if target is None or target in seen else resolve(target, seen + (name,))
    return {a: resolve(a) for a in mapping if resolve(a) != a}


def save_graph(G: nx.DiGraph):
    with open(GRAPH_PATH, "w") as f:
        json.dump(nx.node_link_data(G, edges="edges"), f, indent=1)


def load_graph() -> nx.DiGraph:
    with open(GRAPH_PATH) as f:
        return nx.node_link_graph(json.load(f), edges="edges")


# stats

def print_stats(G: nx.DiGraph, top: int = 15):
    print(f"\nNodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}")

    print("\nTypes:")
    for t, n in Counter(d["type"] for _, d in G.nodes(data=True)).most_common():
        print(f"  {t:12s} {n}")

    print(f"\nMost connected entities:")
    for name, deg in sorted(G.degree, key=lambda x: -x[1])[:top]:
        print(f"  {deg:3d}  {name}")

    print(f"\nEntities shared by the most papers:")
    shared = sorted(G.nodes(data=True), key=lambda x: -len(x[1]["papers"]))
    for name, d in shared[:top]:
        print(f"  {len(d['papers']):2d} papers  {name}")
    n_cross = sum(1 for _, d in G.nodes(data=True) if len(d["papers"]) > 1)
    print(f"  ({n_cross} entities appear in more than one paper)")

    print(f"\nPossible duplicates (similar names):")
    names = sorted(G.nodes)
    pairs = [
        (a, b, SequenceMatcher(None, a, b).ratio())
        for i, a in enumerate(names) for b in names[i + 1:]
    ]
    for a, b, score in sorted((p for p in pairs if p[2] >= 0.85), key=lambda p: -p[2])[:top]:
        print(f"  {score:.2f}  {a!r} ~ {b!r}")


# chunk loading

def load_chunks() -> dict[str, str]:
    """All chunks from the FAISS docstore, keyed by docstore id"""
    docstore = load_vectorstore().docstore._dict
    return {cid: doc.page_content for cid, doc in docstore.items()}


def sample_chunks(chunks: dict[str, str], n: int, seed: int = 0) -> dict[str, str]:
    """n chunks, each from a different paper when possible"""
    by_paper = {}
    for cid, text in chunks.items():
        paper = text.split("\n", 1)[0]          # "Paper: <title>"
        by_paper.setdefault(paper, []).append(cid)
    rng    = random.Random(seed)
    papers = rng.sample(sorted(by_paper), min(n, len(by_paper)))
    return {cid: chunks[cid] for cid in (rng.choice(by_paper[p]) for p in papers)}


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Preview extraction on N chunks (nothing saved)")
    parser.add_argument("--build",  action="store_true",     help="Extract all chunks (cached) and build the graph")
    parser.add_argument("--stats",  action="store_true",     help="Print stats of the saved graph")
    parser.add_argument("--merge-synonyms", action="store_true", help="Find and merge synonym entities (LLM-confirmed)")
    args = parser.parse_args()

    if args.build or args.merge_synonyms:
        chunks      = load_chunks()
        extractions = load_extractions()
        missing     = {cid: t for cid, t in chunks.items() if cid not in extractions}
        print(f"{len(extractions)} chunks cached, extracting {len(missing)}")
        extractions.update(extract_all(missing))
        save_extractions(extractions)
        extractions = {cid: g for cid, g in extractions.items() if cid in chunks}

        if args.merge_synonyms:
            G0    = build_graph(extractions, chunks)
            pairs = synonym_candidates(G0)
            print(f"{len(pairs)} candidate pairs, confirming with the LLM")
            mapping = synonym_map(G0, confirm_synonyms(G0, pairs))
            save_synonyms(mapping)
            print(f"\n{len(mapping)} entities merged into a synonym:")
            groups = defaultdict(list)
            for alias, canonical in mapping.items():
                groups[canonical].append(alias)
            for canonical, aliases in sorted(groups.items()):
                print(f"  {canonical!r} <- {sorted(aliases)}")

        G = build_graph(extractions, chunks, apply_overrides(load_synonyms()))
        save_graph(G)
        print(f"Graph saved → {GRAPH_PATH}")
        print_stats(G)

    if args.stats:
        print_stats(load_graph())

    if args.sample is None:
        raise SystemExit

    sample  = sample_chunks(load_chunks(), args.sample)
    results = extract_all(sample)

    for cid, graph in results.items():
        header = sample[cid].split("\n\n", 1)[0].replace("\n", " | ")
        print(f"\n=== {header}")
        for e in graph.entities:
            print(f"  [{e.type}] {e.name} — {e.description}")
        for r in graph.relations:
            print(f"  {r.source} --{r.relation}--> {r.target}")
