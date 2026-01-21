from nhsea.generators_v3 import EDGE_TOKEN, V3Config, generate_v3_instance


def _parse_prop_id(tok: str) -> int:
    if not tok.startswith("P"):
        raise ValueError(f"Not a prop token: {tok}")
    return int(tok[1:])


def _extract_edges(tokens: list[str]) -> list[tuple[int, int]]:
    edges = []
    idx = 0
    while idx < len(tokens):
        if tokens[idx] == EDGE_TOKEN:
            if idx + 2 >= len(tokens):
                raise ValueError("Truncated edge encoding")
            src = _parse_prop_id(tokens[idx + 1])
            dst = _parse_prop_id(tokens[idx + 2])
            edges.append((src, dst))
            idx += 3
        else:
            idx += 1
    return edges


def _has_cycle(n_nodes: int, edges: list[tuple[int, int]]) -> bool:
    adj = {i: set() for i in range(n_nodes)}
    for src, dst in edges:
        adj[src].add(dst)
    state = [0] * n_nodes

    def _dfs(node: int) -> bool:
        state[node] = 1
        for nxt in adj.get(node, set()):
            if state[nxt] == 1:
                return True
            if state[nxt] == 0 and _dfs(nxt):
                return True
        state[node] = 2
        return False

    for n in range(n_nodes):
        if state[n] == 0 and _dfs(n):
            return True
    return False


def _chain_conclusion(edges: list[tuple[int, int]], premises: list[int], steps: int) -> int:
    adj = {src: set() for src, _ in edges}
    for src, dst in edges:
        adj.setdefault(src, set()).add(dst)
    # Premises point to the same chain start.
    targets = [next(iter(adj[prem])) for prem in premises]
    assert targets[0] == targets[1]
    node = targets[0]
    for _ in range(steps):
        outs = sorted(adj.get(node, set()))
        assert len(outs) == 1
        node = outs[0]
    return node


def test_v3_determinism_same_seed_same_instance():
    cfg = V3Config()
    inst1 = generate_v3_instance("run", "i0", "pair0", "OBC", "conclusion", cfg)
    inst2 = generate_v3_instance("run", "i0", "pair0", "OBC", "conclusion", cfg)
    assert inst1.tokens == inst2.tokens
    assert inst1.candidates == inst2.candidates
    assert inst1.true_index == inst2.true_index


def test_v3_candidate_positions():
    cfg = V3Config()
    inst = generate_v3_instance("run", "i1", "pair1", "OBC", "conclusion", cfg)
    assert len(inst.tokens) == cfg.T
    assert inst.tokens[cfg.T - 2].startswith("P")
    assert inst.tokens[cfg.T - 1].startswith("P")


def test_v3_conclusion_matches_chain_end():
    cfg = V3Config()
    for topology in ("OBC", "PBC"):
        inst = generate_v3_instance("run", f"i2_{topology}", "pair2", topology, "conclusion", cfg)
        edges = _extract_edges(inst.tokens)
        conclusion = _chain_conclusion(edges, inst.premises, cfg.K)
        assert inst.candidates[inst.true_index] == conclusion


def test_v3_topology_cycle_detection():
    cfg = V3Config()
    inst_obc = generate_v3_instance("run", "i3", "pair3", "OBC", "topology", cfg)
    inst_pbc = generate_v3_instance("run", "i4", "pair4", "PBC", "topology", cfg)
    edges_obc = _extract_edges(inst_obc.tokens)
    edges_pbc = _extract_edges(inst_pbc.tokens)
    assert not _has_cycle(cfg.M, edges_obc)
    assert _has_cycle(cfg.M, edges_pbc)
