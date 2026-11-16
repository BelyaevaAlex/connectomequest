"""Exhaustive small typed subgraphs, enumerated from PUBLIC topology only."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .data import canonical_type, ontology_type


@dataclass(frozen=True)
class QueryTemplate:
    name: str
    family: str
    role_types: tuple[str, ...]
    directed_edges: tuple[tuple[int, int], ...]
    source_witnesses: int | None
    original_intents: tuple[str,...] = ()


# Frozen from hemibrain:v1.2.1, threshold5, before any target scoring.
# Source counts are typed-neuron assignments, before Boolean canonicalization.
# Olfactory and MB entries are convergence (role0 -> role1 <- role2).
SOURCE_TEMPLATES = (
    QueryTemplate("olfactory-1","olfactory",("DA1_lPN","LHAV7b1","MBON20"),((0,1),(2,1)),17),
    QueryTemplate("olfactory-2","olfactory",("DA1_lPN","LHAV2a3","MBON20"),((0,1),(2,1)),13),
    QueryTemplate("olfactory-3","olfactory",("DC3_adPN","LHPD2a2","MBON12"),((0,1),(2,1)),12),
    QueryTemplate("olfactory-4","olfactory",("DC3_adPN","LHPD2a2","MBON13"),((0,1),(2,1)),12),
    QueryTemplate("mushroom-1","mushroom",("KCab-m","MBON02","PAM04_a"),((0,1),(2,1)),18716),
    QueryTemplate("mushroom-2","mushroom",("KCg-m","MBON01","PAM01_a"),((0,1),(2,1)),13524),
    QueryTemplate("mushroom-3","mushroom",("KCab-s","MBON02","PAM04_a"),((0,1),(2,1)),12767),
    QueryTemplate("mushroom-4","mushroom",("KCg-m","MBON09","PAM12"),((0,1),(2,1)),12390),
    QueryTemplate("CX-1","CX",("EPG","PEN_a","EPG"),((0,1),(1,2)),4442),
    QueryTemplate("CX-2","CX",("EPG","PEN_b","EPG"),((0,1),(1,2)),2283),
    QueryTemplate("CX-3","CX",("EPG","PEG","EPG"),((0,1),(1,2)),187),
)


def source_roster():
    """Eight unique supported group questions; all11 intents remain audited."""
    merged={}
    for template in SOURCE_TEMPLATES:
        if template.name in {"olfactory-3","olfactory-4"}:
            continue  # Semantic unsupported, explicitly retained in intent_audit.
        roles=tuple(ontology_type(x,"hemibrain:v1.2.1") for x in template.role_types)
        key=(template.family,roles,template.directed_edges)
        if key in merged:
            prior=merged[key]
            merged[key]=replace(prior,original_intents=prior.original_intents+(template.name,),source_witnesses=None)
        else:
            merged[key]=replace(template,role_types=roles,original_intents=(template.name,),
                                source_witnesses=template.source_witnesses if roles==template.role_types else None)
    return tuple(merged.values())


def intent_audit():
    """Preserve original questions, semantic failures and merge provenance."""
    active={name:t.name for t in source_roster() for name in t.original_intents}
    return tuple({"original_intent":t.name,"original_role_types":t.role_types,
                  "status":"supported_common_group" if t.name in active else "semantically_unsupported",
                  "common_template":active.get(t.name),
                  "reason":None if t.name in active else "LHPD2a2 crosswalk includes ambiguous CB0339,CB3774; CB0339 maps LHPD2b1"}
                 for t in SOURCE_TEMPLATES)


def structural_scope(graph, anchor: int, *, radius=1, max_nodes=None):
    """Undirected ego scope from anatomy. Oversized scopes raise; never truncate.

    Anchor choice must be declared independently of private labels. This helper
    does not select biologically interesting anchors or inspect provisional hints.
    """
    if radius < 0 or anchor not in set(map(int,graph.node_ids)):
        raise ValueError("invalid structural anchor/radius")
    incoming = {int(i):set() for i in graph.node_ids}
    for i in graph.node_ids:
        for j in graph.neighbors(int(i)):
            incoming[j].add(int(i))
    seen, frontier = {anchor},{anchor}
    for _ in range(radius):
        nxt=set()
        for cell in frontier:
            nxt.update(graph.neighbors(cell));nxt.update(incoming[cell])
        frontier=nxt-seen;seen.update(nxt)
        if max_nodes is not None and len(seen)>max_nodes:
            raise ValueError("structural scope exceeds cap; retain as computationally unsupported")
    return tuple(sorted(seen))


def enumerate_typed_query(graph, query_id, role_types, directed_edges, *, candidate_ids=None,
                          fixed_roles=None, max_witnesses=None, max_search_states=None):
    """All distinct-cell assignments satisfying the declared directed topology.

    Labels (including provisional labels) NEVER restrict candidate assignments.
    Exhaustion caps retain an explicit unsupported Query with no partial family.
    Consumers MUST reject computationally_unsupported before calling evaluate;
    an unsupported query is not an empty, negative biological query.
    """
    from .core import Witness, canonical_query
    labels=tuple(graph.label_code(canonical_type(name)) for name in role_types)
    if 0 in labels:
        raise ValueError("OTHER is a residual oracle code, not a biological query predicate")
    n=len(labels)
    if n < 1:
        raise ValueError("at least one typed role is required")
    edges=tuple(sorted(set(directed_edges)))
    if any(a<0 or b<0 or a>=n or b>=n or a==b for a,b in edges):
        raise ValueError("invalid pattern edge")
    reach={0}
    while True:
        nxt=reach|{b for a,b in edges if a in reach}|{a for a,b in edges if b in reach}
        if nxt==reach:break
        reach=nxt
    if len(reach)!=n:
        raise ValueError("pattern must be connected")
    cells=tuple(sorted(set(map(int,graph.node_ids if candidate_ids is None else candidate_ids))))
    if not set(cells)<=set(map(int,graph.node_ids)):
        raise ValueError("candidate scope contains unknown cells")
    domain=set(cells)
    outgoing_cache={};incoming_cache={}
    def outgoing(cell):
        if cell not in outgoing_cache:outgoing_cache[cell]=set(graph.neighbors(cell))&domain
        return outgoing_cache[cell]
    def incoming(cell):
        if cell not in incoming_cache:incoming_cache[cell]=set(graph.predecessors(cell))&domain
        return incoming_cache[cell]
    # Begin at the highest-degree pattern role, then extend along known edges.
    degree={i:sum(i in edge for edge in edges) for i in range(n)}
    fixed={int(role):int(cell) for role,cell in (fixed_roles or {}).items()}
    if any(role<0 or role>=n or cell not in domain for role,cell in fixed.items()) or len(set(fixed.values()))!=len(fixed):
        raise ValueError("invalid fixed roles")
    order=sorted(fixed)
    if not order:
        order=[min(range(n),key=lambda i:(-degree[i],i))]
    while len(order)<n:
        remain=set(range(n))-set(order)
        order.append(min(remain,key=lambda i:(-sum((i,j) in edges or (j,i) in edges for j in order),-degree[i],i)))
    metadata={"role_types":list(map(canonical_type,role_types)),"directed_edges":[list(x) for x in edges],
              "candidate_ids":list(cells),"enumeration_complete":True,"computationally_unsupported":False,
              "fixed_roles":dict(fixed),"scope":"explicit public candidate universe","topology_only_enumeration":True}
    order=[role for role in order if role not in fixed]
    witnesses=[];assigned=dict(fixed);used=set(fixed.values());states=0;unsupported=False
    def search(depth):
        nonlocal states,unsupported
        if depth==len(order):
            vertices=tuple(assigned[i] for i in range(n))
            witnesses.append(Witness(query_id+":"+",".join(map(str,vertices)),tuple(zip(vertices,labels)),vertices))
            if max_witnesses is not None and len(witnesses)>max_witnesses:unsupported=True
            return
        role=order[depth];possible=None
        for other,cell in assigned.items():
            needed=[]
            if (other,role) in edges:needed.append(outgoing(cell))
            if (role,other) in edges:needed.append(incoming(cell))
            for candidate in needed:
                possible=set(candidate) if possible is None else possible&candidate
        for cell in sorted(domain if possible is None else possible):
            if cell in used:continue
            states+=1
            if max_search_states is not None and states>max_search_states:
                unsupported=True;return
            assigned[role]=cell;used.add(cell);search(depth+1);used.remove(cell);del assigned[role]
            if unsupported:return
    if all(a not in fixed or b not in fixed or fixed[b] in outgoing(fixed[a]) for a,b in edges):
        search(0)
    metadata["search_states"]=states
    if unsupported:
        metadata.update(enumeration_complete=False,computationally_unsupported=True,
                        unsupported_reason="exact enumeration exceeded declared computational cap")
        witnesses=[]
    else:
        metadata["raw_assignments"]=len(witnesses)
    return canonical_query(query_id,witnesses,metadata)
