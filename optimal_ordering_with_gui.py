#!/usr/bin/env python3
"""Epitaxy Deposition Planner — paste strips, dial effort, get a schedule.

Finds a physical arrangement of substrate strips and a minimum-step
deposition schedule (element + contiguous mask window per step) such that
every strip receives its recipe layers in order.

USAGE
    python3 deposition_planner.py                  # GUI (falls back to paste mode)
    python3 deposition_planner.py strips.txt       # read strips from a file
    python3 deposition_planner.py strips.txt -e 4  # more effort = more optimality
    python3 deposition_planner.py -o plan.txt      # also save the schedule to a file
    cat strips.txt | python3 deposition_planner.py --cli

INPUT FORMATS (auto-detected)
    1. Python dict, as copied from a notebook:
           test_genomes = { 1: 'Mn-Co-Al-Co-...', 2: 'Mn-Fe-...', }
    2. One strip per line, with or without an id:
           1: Mn-Co-Al-Co
           Mn Fe Al Co        (separators: '-', ',' or spaces)
    Lines starting with '#' are ignored.

EFFORT DIAL (default 3)
    1  quick     ~seconds     good schedule, no optimality proof
    2  fast      ~10 s        near-optimal, proof for small sets
    3  standard  ~30-60 s     provably optimal for <= ~20 strips (default)
    4  thorough  ~2-3 min     proof reach ~22 strips, deeper search beyond
    5  maximum   ~10-15 min   everything the machine can give

All schedules are checked by an independent verifier before being reported;
the tool never returns a physically invalid schedule.
"""

import argparse
import math
import multiprocessing as _mp
import os
import random
import re
import sys
import time
from collections import defaultdict
from itertools import combinations

try:
    import numpy as np
except ImportError:
    np = None

# ============================================================================
# Input parsing
# ============================================================================

_ELEM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,2}$")


def parse_strips(text):
    """Parse pasted strip recipes. Returns {id: [element, ...]} (ordered)."""
    quoted = re.findall(r"(?:(\d+)\s*:\s*)?['\"]([A-Za-z][A-Za-z0-9,\- ]*)['\"]",
                        text)
    strips = {}
    if quoted and any(q[1].count('-') or ' ' in q[1] or ',' in q[1]
                      for q in quoted):
        auto = 1
        for sid, body in quoted:
            elems = [t for t in re.split(r"[-,\s]+", body.strip()) if t]
            if not elems or not all(_ELEM_RE.match(e) for e in elems):
                raise ValueError(f"could not read recipe: {body!r}")
            key = int(sid) if sid else auto
            while key in strips:
                key += 1
            strips[key] = elems
            auto = key + 1
        return strips

    auto = 1
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        m = re.match(r"^(\d+)\s*[:,]\s*(.+)$", line)
        sid, body = (int(m.group(1)), m.group(2)) if m else (None, line)
        elems = [t for t in re.split(r"[-,\s]+", body.strip()) if t]
        if not elems or not all(_ELEM_RE.match(e) for e in elems):
            raise ValueError(f"could not read line: {raw!r}")
        key = sid if sid is not None else auto
        while key in strips:
            key += 1
        strips[key] = elems
        auto = key + 1
    if not strips:
        raise ValueError("no strip recipes found in the input")
    return strips


# ============================================================================
# Core: verifier, metrics, LCS utilities
# ============================================================================

def verify(seqs, schedule):
    """Validity + completeness check. seqs: list of recipes in slot order."""
    n = len(seqs)
    ptr = [0] * n
    for t, (elem, i, j) in enumerate(schedule):
        if not (0 <= i <= j < n):
            return False, f"step {t + 1}: window [{i},{j}] out of range"
        for k in range(i, j + 1):
            if ptr[k] >= len(seqs[k]):
                return False, f"step {t + 1}: slot {k} already complete"
            if seqs[k][ptr[k]] != elem:
                return False, (f"step {t + 1}: slot {k} needs "
                               f"{seqs[k][ptr[k]]}, exposed to {elem}")
            ptr[k] += 1
    for k in range(n):
        if ptr[k] != len(seqs[k]):
            return False, f"slot {k} incomplete ({ptr[k]}/{len(seqs[k])})"
    return True, "valid and complete"


def lcs_suffix_table(a, b):
    """T[i][j] = LCS length of a[i:] and b[j:]."""
    la, lb = len(a), len(b)
    T = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la - 1, -1, -1):
        Ti, Ti1 = T[i], T[i + 1]
        ai = a[i]
        for j in range(lb - 1, -1, -1):
            if ai == b[j]:
                Ti[j] = Ti1[j + 1] + 1
            else:
                x, y = Ti1[j], Ti[j + 1]
                Ti[j] = x if x >= y else y
    return T


def lcs_len(a, b):
    return lcs_suffix_table(a, b)[0][0]


def lcs_match(A, B):
    """(LCS length, one concrete matching as [(i, j), ...])."""
    dp = lcs_suffix_table(A, B)
    match = []
    i = j = 0
    while i < len(A) and j < len(B):
        if A[i] == B[j] and dp[i][j] == dp[i + 1][j + 1] + 1:
            match.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return dp[0][0], match


# ============================================================================
# Method 1: LCS route — Held-Karp / 2-opt ordering + alignment scheduler
# ============================================================================

def _held_karp_path(W):
    n = len(W)
    size = 1 << n
    NEG = -(1 << 12)
    Wt = np.asarray(W, dtype=np.int16)
    dp = np.full((size, n), NEG, dtype=np.int16)
    for i in range(n):
        dp[1 << i, i] = 0
    masks = np.arange(size, dtype=np.int64)
    pc = np.zeros(size, dtype=np.uint8)
    for b in range(n):
        pc += ((masks >> b) & 1).astype(np.uint8)
    by_pc = np.argsort(pc, kind="stable")
    bounds = np.searchsorted(pc[by_pc], np.arange(n + 2))
    for k in range(1, n):
        layer = by_pc[bounds[k]:bounds[k + 1]]
        for j in range(n):
            bit = 1 << j
            src = layer[(layer & bit) == 0]
            if src.size:
                dp[src + bit, j] = (dp[src] + Wt[:, j]).max(axis=1)
    full = size - 1
    j = int(np.argmax(dp[full]))
    order, mask = [j], full
    while mask != (1 << j):
        pm = mask ^ (1 << j)
        row = dp[pm]
        best_i, best_v = -1, None
        for i in range(n):
            if (pm >> i) & 1:
                v = int(row[i]) + W[i][j]
                if best_v is None or v > best_v:
                    best_i, best_v = i, v
        order.append(best_i)
        mask, j = pm, best_i
    order.reverse()
    return order


def _path_weight(order, W):
    return sum(W[order[k]][order[k + 1]] for k in range(len(order) - 1))


def _local_search_path(W):
    n = len(W)
    best = None
    for s in range(n):
        left, cur, path = set(range(n)) - {s}, s, [s]
        while left:
            nxt = max(left, key=lambda x: W[cur][x])
            path.append(nxt)
            left.remove(nxt)
            cur = nxt
        if best is None or _path_weight(path, W) > _path_weight(best, W):
            best = path
    improved = True
    while improved:
        improved = False
        for a in range(n - 1):
            for b in range(a + 1, n):
                cand = best[:a] + best[a:b + 1][::-1] + best[b + 1:]
                if _path_weight(cand, W) > _path_weight(best, W):
                    best, improved = cand, True
    return best


class _SchedCtx:
    """Scheduling context for a fixed arrangement, with suffix-LCS bound."""

    def __init__(self, seqs):
        self.seqs = seqs
        self.n = len(seqs)
        self.lens = [len(s) for s in seqs]
        self.goal = tuple(self.lens)
        self.H = [lcs_suffix_table(seqs[k], seqs[k + 1])
                  for k in range(self.n - 1)]

    def h(self, ptr):
        rem = sum(self.lens[k] - ptr[k] for k in range(self.n))
        sav = sum(self.H[k][ptr[k]][ptr[k + 1]] for k in range(self.n - 1))
        return rem - sav

    def moves(self, ptr):
        seqs, lens, n = self.seqs, self.lens, self.n
        out = []
        k = 0
        while k < n:
            if ptr[k] >= lens[k]:
                k += 1
                continue
            e = seqs[k][ptr[k]]
            a = k
            while k + 1 < n and ptr[k + 1] < lens[k + 1] \
                    and seqs[k + 1][ptr[k + 1]] == e:
                k += 1
            out.append((e, a, k))
            for x in range(a, k):
                out.append((e, a, x))
            for y in range(a + 1, k + 1):
                out.append((e, y, k))
            k += 1
        return out


def _apply(ptr, mv):
    _, i, j = mv
    lst = list(ptr)
    for k in range(i, j + 1):
        lst[k] += 1
    return tuple(lst)


def _greedy_schedule(ctx, prefer_widest):
    ptr = (0,) * ctx.n
    sched = []
    while ptr != ctx.goal:
        best_key = best_mv = best_ptr = None
        for mv in ctx.moves(ptr):
            nptr = _apply(ptr, mv)
            hv = ctx.h(nptr)
            w = mv[2] - mv[1]
            key = (-w, hv) if prefer_widest else (hv, -w)
            if best_key is None or key < best_key:
                best_key, best_mv, best_ptr = key, mv, nptr
        sched.append(best_mv)
        ptr = best_ptr
    return sched


def _beam_over_states(ctx, width, best_cost, deadline):
    start = (0,) * ctx.n
    if start == ctx.goal:
        return 0, []
    frontier = [(ctx.h(start), start, None)]
    best_node, g = None, 0
    while frontier:
        if time.perf_counter() > deadline:
            break
        g += 1
        nxt = {}
        for _, ptr, node in frontier:
            for mv in ctx.moves(ptr):
                nptr = _apply(ptr, mv)
                if nptr == ctx.goal:
                    if g < best_cost:
                        best_cost, best_node = g, (mv, node)
                    continue
                if nptr in nxt:
                    continue
                f = g + ctx.h(nptr)
                if f < best_cost:
                    nxt[nptr] = (f, nptr, (mv, node))
        if not nxt:
            break
        frontier = sorted(nxt.values(), key=lambda x: x[0])[:width]
    if best_node is None:
        return None, None
    sched, node = [], best_node
    while node is not None:
        sched.append(node[0])
        node = node[1]
    sched.reverse()
    return best_cost, sched


def method_lcs_route(ids, seqs_by_id, budget_s, hk_cutoff):
    """Returns (order, schedule, info). info: path_weight, hk (exactness)."""
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    n = len(ids)
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            W[i][j] = W[j][i] = lcs_len(seqs_by_id[ids[i]], seqs_by_id[ids[j]])
    use_hk = n <= hk_cutoff and np is not None
    base = _held_karp_path(W) if use_hk else _local_search_path(W)
    best_w = _path_weight(base, W)

    def canon(o):
        t = tuple(o)
        return min(t, t[::-1])

    cands, seen = [base], {canon(base)}
    for a in range(n - 1):
        for b in range(a + 1, n):
            nb = base[:a] + base[a:b + 1][::-1] + base[b + 1:]
            if _path_weight(nb, W) == best_w and canon(nb) not in seen:
                seen.add(canon(nb))
                cands.append(nb)
                if len(cands) >= 12:
                    break
        if len(cands) >= 12:
            break

    best, scored, ctxs = None, [], {}
    for ci, cand in enumerate(cands):
        order_ids = [ids[i] for i in cand]
        ctx = _SchedCtx([seqs_by_id[g] for g in order_ids])
        ctxs[ci] = (order_ids, ctx)
        cand_best = None
        for prefer_widest in (False, True):
            sched = _greedy_schedule(ctx, prefer_widest)
            if best is None or len(sched) < best[0]:
                best = (len(sched), order_ids, sched)
            if cand_best is None or len(sched) < cand_best:
                cand_best = len(sched)
        scored.append((cand_best, ci))
        if time.perf_counter() > deadline - budget_s * 0.2:
            break
    scored.sort()
    for width in (64, 256, 1024):
        for _, ci in scored[:3]:
            if time.perf_counter() > deadline:
                break
            order_ids, ctx = ctxs[ci]
            cost, sched = _beam_over_states(ctx, width, best[0], deadline)
            if sched is not None and cost < best[0]:
                best = (cost, order_ids, sched)
    _, order_ids, sched = best
    return order_ids, sched, {"path_weight": best_w, "hk": use_hk}


# ============================================================================
# Method 2: simulated annealing over arrangements + LCS-merge scheduler
# ============================================================================

def _try_topo(seqs, offs, matches):
    total = offs[-1]
    parent = list(range(total))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, mlist in enumerate(matches):
        for a, b in mlist:
            ra, rb = find(offs[k] + a), find(offs[k + 1] + b)
            if ra != rb:
                parent[ra] = rb
    members = defaultdict(list)
    slot_of = [0] * total
    for k in range(len(seqs)):
        for p in range(len(seqs[k])):
            node = offs[k] + p
            slot_of[node] = k
            members[find(node)].append(node)
    indeg, adj = defaultdict(int), defaultdict(list)
    self_loop = False
    for k in range(len(seqs)):
        for p in range(len(seqs[k]) - 1):
            u, v = find(offs[k] + p), find(offs[k] + p + 1)
            if u == v:
                self_loop = True
            adj[u].append(v)
            indeg[v] += 1
    schedule, done = [], set()
    if not self_loop:
        queue = [g for g in members if indeg[g] == 0]
        while queue:
            g = queue.pop()
            done.add(g)
            nodes = members[g]
            k0 = slot_of[nodes[0]]
            elem = seqs[k0][nodes[0] - offs[k0]]
            slots = [slot_of[nd] for nd in nodes]
            schedule.append((elem, min(slots), max(slots)))
            for v in adj[g]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
    if len(done) == len(members):
        return schedule, None
    for k, mlist in enumerate(matches):
        for idx, (a, b) in enumerate(mlist):
            if find(offs[k] + a) not in done:
                return None, (k, idx)
    raise RuntimeError("cycle with no removable match")


def build_merge_schedule(order, seqs_by_id, match_cache):
    seqs = [seqs_by_id[g] for g in order]
    offs = [0]
    for s in seqs:
        offs.append(offs[-1] + len(s))
    matches = [list(match_cache[(order[k], order[k + 1])])
               for k in range(len(order) - 1)]
    while True:
        schedule, stuck = _try_topo(seqs, offs, matches)
        if schedule is not None:
            return schedule
        k, idx = stuck
        matches[k].pop(idx)


def _sa_score(order, W2):
    s, prev = 0, order[0]
    for g in order[1:]:
        s += W2[prev][g]
        prev = g
    return s


def _sa_greedy_order(ids, W2, rng):
    if len(ids) < 2:
        return list(ids)
    best = max(((W2[a][b], a, b) for a in ids for b in ids if a != b))
    path = [best[1], best[2]]
    remaining = set(ids) - set(path)
    while remaining:
        cands = []
        for g in remaining:
            cands.append((W2[path[0]][g], rng.random(), g, True))
            cands.append((W2[path[-1]][g], rng.random(), g, False))
        _, _, g, front = max(cands)
        path.insert(0, g) if front else path.append(g)
        remaining.discard(g)
    return path


def _anneal_once(start, W2, rng, deadline, iters):
    cur = list(start)
    score = _sa_score(cur, W2)
    best, bscore = cur[:], score
    n = len(cur)
    T0, Tmin = 4.0, 0.05
    alpha = (Tmin / T0) ** (1.0 / iters)
    T = T0
    for it in range(iters):
        if (it & 255) == 0 and time.time() > deadline:
            break
        r = rng.random()
        i, j = rng.randrange(n), rng.randrange(n)
        new = cur[:]
        if r < 0.45:
            if i > j:
                i, j = j, i
            new[i:j + 1] = new[i:j + 1][::-1]
        elif r < 0.75:
            new[i], new[j] = new[j], new[i]
        else:
            g = new.pop(i)
            new.insert(rng.randrange(n), g)
        s = _sa_score(new, W2)
        d = s - score
        if d >= 0 or rng.random() < math.exp(d / T):
            cur, score = new, s
            if score > bscore:
                best, bscore = cur[:], score
        T *= alpha
    return best, bscore


def _sa_search(ids, W2, rng, deadline, ub, greedy_seed):
    n = len(ids)
    iters = max(2000, 2500 * n)
    best, bscore = None, -1
    restarts = since = 0
    while time.time() < deadline:
        if greedy_seed and restarts == 0:
            start = _sa_greedy_order(ids, W2, rng)
        else:
            start = list(ids)
            rng.shuffle(start)
        order, score = _anneal_once(start, W2, rng, deadline, iters)
        restarts += 1
        if score > bscore:
            best, bscore, since = order, score, 0
        else:
            since += 1
        if bscore >= ub or (restarts >= 25 and since >= 20):
            break
    return bscore, best


def _sa_worker(ids, W2, seed, deadline, ub, greedy_seed, q):
    try:
        q.put(_sa_search(ids, W2, random.Random(seed), deadline, ub,
                         greedy_seed))
    except Exception:
        q.put((-1, None))


def method_anneal(ids, seqs_by_id, budget_s, workers=3):
    t0 = time.time()
    ids = list(ids)
    n = len(ids)
    W2 = {a: {} for a in ids}
    match_cache = {}
    for a, b in combinations(ids, 2):
        l, m = lcs_match(seqs_by_id[a], seqs_by_id[b])
        W2[a][b] = W2[b][a] = l
        match_cache[(a, b)] = m
        match_cache[(b, a)] = [(y, x) for x, y in m]
    pair_ws = sorted((W2[a][b] for a, b in combinations(ids, 2)), reverse=True)
    ub = sum(pair_ws[:n - 1])
    deadline = t0 + budget_s

    results = []
    if n >= 4 and workers > 1:
        try:
            ctx = _mp.get_context("fork")
            q = ctx.Queue()
            procs = []
            for w in range(workers):
                p = ctx.Process(target=_sa_worker,
                                args=(ids, W2, 12345 + 7919 * w, deadline,
                                      ub, w == 0, q))
                p.start()
                procs.append(p)
            for _ in procs:
                try:
                    results.append(
                        q.get(timeout=max(0.5, deadline - time.time()) + 3.0))
                except Exception:
                    break
            for p in procs:
                p.join(timeout=1.0)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1.0)
        except Exception:
            results = []
    if not results:
        results.append(_sa_search(ids, W2, random.Random(1), deadline, ub,
                                  True))
    bscore, best = max(results, key=lambda r: r[0])
    if best is None:
        best = list(ids)
    schedule = build_merge_schedule(best, seqs_by_id, match_cache)
    return best, schedule, {"score": bscore, "pair_ub": ub}


# ============================================================================
# Method 3: clustered seeds + beam search over schedules
# ============================================================================

def _agglomerative_order(ids, sim, reverse_ties=False):
    clusters = [[g] for g in sorted(ids, reverse=reverse_ties)]
    while len(clusters) > 1:
        best = None
        for x in range(len(clusters)):
            cx = clusters[x]
            for y in range(x + 1, len(clusters)):
                cy = clusters[y]
                for ex_head in (False, True):
                    ex = cx[0] if ex_head else cx[-1]
                    for ey_tail in (False, True):
                        ey = cy[-1] if ey_tail else cy[0]
                        s = sim[ex][ey]
                        if best is None or s > best[0]:
                            best = (s, x, y, ex_head, ey_tail)
        _, x, y, ex_head, ey_tail = best
        cx, cy = clusters[x], clusters[y]
        if ex_head:
            cx = cx[::-1]
        if ey_tail:
            cy = cy[::-1]
        merged = cx + cy
        clusters = [c for k, c in enumerate(clusters) if k not in (x, y)]
        clusters.append(merged)
    return clusters[0]


def _nn_chain_order(ids, sim):
    best = None
    for start in ids:
        rest = [g for g in ids if g != start]
        chain = [start]
        while rest:
            tail = chain[-1]
            nxt = max(rest, key=lambda g: sim[tail][g])
            chain.append(nxt)
            rest.remove(nxt)
        s = sum(sim[chain[k]][chain[k + 1]] for k in range(len(chain) - 1))
        if best is None or s > best[0]:
            best = (s, chain)
    return best[1]


def _two_opt_ids(order, sim):
    order = list(order)
    n = len(order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                delta = 0
                if i > 0:
                    delta += sim[order[i - 1]][order[j]] - \
                        sim[order[i - 1]][order[i]]
                if j < n - 1:
                    delta += sim[order[i]][order[j + 1]] - \
                        sim[order[j]][order[j + 1]]
                if delta > 0:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
    return order


_FULL_ENUM_RUN, _TRIM_CAP = 4, 2


def _candidate_windows(nxt, n):
    windows = []
    a = 0
    while a < n:
        e = nxt[a]
        if e is None:
            a += 1
            continue
        b = a
        while b + 1 < n and nxt[b + 1] == e:
            b += 1
        L = b - a + 1
        ws = set()
        if L <= _FULL_ENUM_RUN:
            for x in range(a, b + 1):
                for y in range(x, b + 1):
                    ws.add((x, y))
        else:
            for da in range(_TRIM_CAP + 1):
                for db in range(_TRIM_CAP + 1):
                    x, y = a + da, b - db
                    if x <= y:
                        ws.add((x, y))
            for x in range(a, b + 1):
                ws.add((x, x))
        for x, y in ws:
            windows.append((e, x, y))
        a = b + 1
    return windows


def _beam_schedule(order, seqs_by_id, width, deadline):
    seqs = [seqs_by_id[g] for g in order]
    n = len(seqs)
    lens = [len(s) for s in seqs]
    pairT = [lcs_suffix_table(seqs[k], seqs[k + 1]) for k in range(n - 1)]
    start_ptr = (0,) * n
    pv0 = tuple(pairT[k][0][0] for k in range(n - 1))
    records = [(-1, None)]
    beam = [(start_ptr, sum(lens), pv0, 0)]
    visited = {start_ptr}
    while beam:
        cand = {}
        for ptr, sum_rem, pairvals, rec in beam:
            nxt = [seqs[k][ptr[k]] if ptr[k] < lens[k] else None
                   for k in range(n)]
            for e, i, j in _candidate_windows(nxt, n):
                lp = list(ptr)
                for k in range(i, j + 1):
                    lp[k] += 1
                nptr = tuple(lp)
                if nptr in visited or nptr in cand:
                    continue
                nsum = sum_rem - (j - i + 1)
                npv = list(pairvals)
                for k in range(max(0, i - 1), min(j, n - 2) + 1):
                    npv[k] = pairT[k][nptr[k]][nptr[k + 1]]
                h = max(nsum - sum(npv),
                        max(lens[k] - nptr[k] for k in range(n)))
                cand[nptr] = (h, nsum, tuple(npv), rec, (e, i, j))
        entries = sorted(cand.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        w = width if time.time() < deadline else 16
        new_beam = []
        for nptr, (h, nsum, npv, prec, step) in entries[:w]:
            rec_id = len(records)
            records.append((prec, step))
            if nsum == 0:
                sched, r = [], rec_id
                while r:
                    p, s = records[r]
                    sched.append(s)
                    r = p
                sched.reverse()
                return sched
            visited.add(nptr)
            new_beam.append((nptr, nsum, npv, rec_id))
        beam = new_beam
    return None


def _base_beam_width(n):
    for cap, w in ((5, 6000), (8, 3500), (11, 2000), (14, 1200), (17, 800)):
        if n <= cap:
            return w
    return 600


def method_beam(ids, seqs_by_id, budget_s, width_factor=1.0):
    t0 = time.time()
    deadline = t0 + budget_s
    ids = list(ids)
    n = len(ids)
    sim = {a: {b: (lcs_len(seqs_by_id[a], seqs_by_id[b]) if a != b else 0)
               for b in ids} for a in ids}
    raw = [
        _agglomerative_order(ids, sim),
        _two_opt_ids(_agglomerative_order(ids, sim), sim),
        _two_opt_ids(_agglomerative_order(ids, sim, reverse_ties=True), sim),
        _two_opt_ids(_nn_chain_order(ids, sim), sim),
    ]
    seeds, seen = [], set()
    for o in raw:
        key = min(tuple(o), tuple(reversed(o)))
        if key not in seen:
            seen.add(key)
            seeds.append(o)
    seeds.sort(key=lambda o: -sum(sim[o[k]][o[k + 1]]
                                  for k in range(len(o) - 1)))
    width = max(32, int(_base_beam_width(n) * width_factor))
    best = None
    for idx, order in enumerate(seeds):
        now = time.time()
        if best is not None and now >= deadline:
            break
        slot = (deadline - now) / (len(seeds) - idx)
        sched = _beam_schedule(order, seqs_by_id, width, now + slot)
        if sched is not None and (best is None or len(sched) < len(best[1])):
            best = (order, sched)
    return best[0], best[1], {}


# ============================================================================
# Method 4: exact A* + branch-and-bound (run in a killable subprocess)
# ============================================================================

def _astar_fixed(seqs, pair_tabs, ub=None):
    import heapq
    n = len(seqs)
    lens = tuple(len(s) for s in seqs)
    goal = lens
    INF = float("inf")

    def h(state):
        tot = mx = 0
        for k in range(n):
            r = lens[k] - state[k]
            tot += r
            if r > mx:
                mx = r
        sh = sum(pair_tabs[k][state[k]][state[k + 1]] for k in range(n - 1))
        lb = tot - sh
        return lb if lb > mx else mx

    start = (0,) * n
    h0 = h(start)
    if ub is not None and h0 >= ub:
        return None, None
    g_best, came = {start: 0}, {}
    open_heap = [(h0, 0, start)]
    while open_heap:
        f, ng_neg, s = heapq.heappop(open_heap)
        g = -ng_neg
        if g > g_best.get(s, INF):
            continue
        if s == goal:
            sched, cur = [], s
            while cur != start:
                cur, step = came[cur]
                sched.append(step)
            sched.reverse()
            return g, sched
        nexts = [seqs[k][s[k]] if s[k] < lens[k] else None for k in range(n)]
        ng = g + 1
        k = 0
        while k < n:
            e = nexts[k]
            if e is None:
                k += 1
                continue
            m = k
            while m + 1 < n and nexts[m + 1] == e:
                m += 1
            for i in range(k, m + 1):
                for j in range(i, m + 1):
                    ns = list(s)
                    for t in range(i, j + 1):
                        ns[t] += 1
                    ns = tuple(ns)
                    if ng < g_best.get(ns, INF):
                        nf = ng + h(ns)
                        if ub is None or nf < ub:
                            g_best[ns] = ng
                            came[ns] = (s, (e, i, j))
                            heapq.heappush(open_heap, (nf, -ng, ns))
            k = m + 1
    return None, None


def exact_solve(seq_list):
    """Provably optimal (perm_indices, schedule) over arrangements+schedules."""
    n = len(seq_list)
    seq_list = [tuple(s) for s in seq_list]
    total = sum(len(s) for s in seq_list)
    uniq = sorted(set(seq_list))
    tab_cache = {}
    for a in uniq:
        for b in uniq:
            if (a, b) not in tab_cache:
                tab_cache[(a, b)] = lcs_suffix_table(a, b)

    def tabs_for(ps):
        return [tab_cache[(ps[k], ps[k + 1])] for k in range(len(ps) - 1)]

    if n == 1:
        return [0], [(e, 0, 0) for e in seq_list[0]]

    W = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v:
                W[u][v] = tab_cache[(seq_list[u], seq_list[v])][0][0]

    full = (1 << n) - 1
    Wnp = np.array(W, dtype=np.int16)
    f = np.full((1 << n, n), -1, dtype=np.int16)
    for v in range(n):
        f[1 << v, v] = 0
    for mask in range(1, full):
        fm = f[mask]
        vs = [b for b in range(n) if (mask >> b) & 1]
        us = np.array([b for b in range(n) if not (mask >> b) & 1])
        cand = (fm[vs, None] + Wnp[vs]).max(axis=0)
        rows = mask | (1 << us)
        f[rows, us] = np.maximum(f[rows, us], cand[us])
    f_max = int(f[full].max())
    lb_global = max(total - f_max, max(len(s) for s in seq_list))

    v = max(range(n), key=lambda x: f[full][x])
    mask, path = full, [v]
    while mask != (1 << v):
        pm = mask & ~(1 << v)
        for u in range(n):
            if (pm >> u) & 1 and f[pm][u] >= 0 \
                    and f[pm][u] + W[u][v] == f[mask][v]:
                path.append(u)
                mask, v = pm, u
                break
    path.reverse()

    perm_seqs = tuple(seq_list[k] for k in path)
    best_moves, best_sched = _astar_fixed(perm_seqs, tabs_for(perm_seqs))
    best_path = list(path)

    if best_moves > lb_global:
        def key_of(k):
            return seq_list[k]

        def dfs(prefix, mask, share):
            nonlocal best_moves, best_sched, best_path
            v = prefix[-1]
            if mask == full:
                fwd = tuple(key_of(k) for k in prefix)
                if fwd <= fwd[::-1]:
                    moves, sched = _astar_fixed(fwd, tabs_for(fwd),
                                                ub=best_moves)
                    if moves is not None and moves < best_moves:
                        best_moves, best_sched = moves, sched
                        best_path = list(prefix)
                return
            need = total - best_moves
            seen = set()
            r = full & ~mask
            while r:
                low = r & -r
                u = low.bit_length() - 1
                r ^= low
                ku = key_of(u)
                if ku in seen:
                    continue
                seen.add(ku)
                ns = share + W[v][u]
                comp = f[(full & ~(mask | (1 << u))) | (1 << u)][u]
                if ns + comp <= need:
                    continue
                prefix.append(u)
                dfs(prefix, mask | (1 << u), ns)
                prefix.pop()

        starts_seen = set()
        for v in range(n):
            kv = key_of(v)
            if kv in starts_seen:
                continue
            starts_seen.add(kv)
            if f[full][v] > total - best_moves:
                dfs([v], 1 << v, 0)
    return best_path, best_sched


def _exact_worker(seq_list, conn):
    try:
        conn.send(exact_solve(seq_list))
    except Exception:
        try:
            conn.send(None)
        except Exception:
            pass
    finally:
        conn.close()


def method_exact(ids, seqs_by_id, cap_s):
    """Run the exact solver with a hard kill after cap_s seconds."""
    seq_list = [seqs_by_id[g] for g in ids]
    ctx = _mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_exact_worker, args=(seq_list, child), daemon=True)
    p.start()
    child.close()
    result = None
    if parent.poll(cap_s):
        try:
            result = parent.recv()
        except EOFError:
            result = None
    p.join(timeout=0.5)
    if p.is_alive():
        p.terminate()
        p.join(timeout=2.0)
    parent.close()
    if result is None:
        return None
    perm, sched = result
    return [ids[k] for k in perm], sched, {}


# ============================================================================
# Portfolio driver — the effort dial lives here
# ============================================================================

EFFORT_LEVELS = {
    1: dict(label="quick",    lcs_s=1.5,  sa_s=1.0,  beam_s=0,   bw=0.0,
            exact_cap=0,   hk_cutoff=15),
    2: dict(label="fast",     lcs_s=4.0,  sa_s=3.0,  beam_s=0,   bw=0.0,
            exact_cap=4,   hk_cutoff=18),
    3: dict(label="standard", lcs_s=10.0, sa_s=8.0,  beam_s=8,   bw=0.5,
            exact_cap=25,  hk_cutoff=20),
    4: dict(label="thorough", lcs_s=25.0, sa_s=30.0, beam_s=25,  bw=1.0,
            exact_cap=100, hk_cutoff=21),
    5: dict(label="maximum",  lcs_s=60.0, sa_s=90.0, beam_s=90,  bw=2.5,
            exact_cap=600, hk_cutoff=23),
}
_EXACT_MEM_LIMIT_N = 24  # 2^N DP table: keep under ~1 GB


def _predict_exact_seconds(n):
    return 1.3 * (1.9 ** (n - 15))


def plan(seqs_by_id, effort=3, log=lambda s: None):
    """Run the method portfolio at the given effort. Returns a result dict."""
    cfg = EFFORT_LEVELS[max(1, min(5, int(effort)))]
    ids = list(seqs_by_id)
    n = len(ids)
    baseline = sum(len(s) for s in seqs_by_id.values())
    max_len = max(len(s) for s in seqs_by_id.values())
    t_start = time.time()
    log(f"{n} strips, serial baseline {baseline} steps — "
        f"effort {effort} ({cfg['label']})")

    if n == 1:
        g = ids[0]
        sched = [(e, 0, 0) for e in seqs_by_id[g]]
        return dict(order=ids, schedule=sched, moves=len(sched),
                    baseline=baseline, status="PROVEN OPTIMAL",
                    proof="single strip", methods=[], lower_bound=len(sched),
                    wall_s=0.0)

    # Cheap always-available lower bound: any arrangement has n-1 adjacent
    # pairs and each pair shares at most its LCS.
    pair_ws = sorted((lcs_len(seqs_by_id[a], seqs_by_id[b])
                      for a, b in combinations(ids, 2)), reverse=True)
    lb = max(max_len, baseline - sum(pair_ws[:n - 1]))
    lb_exactness = "pair bound"

    best = None          # (moves, order, sched, method)
    proven = None        # proof string once optimality is certain
    methods = []         # per-method report rows

    def consider(name, order, sched, wall):
        nonlocal best
        ok, msg = verify([seqs_by_id[g] for g in order], sched)
        if not ok:
            methods.append((name, None, wall, f"REJECTED: {msg}"))
            log(f"  {name}: rejected by verifier ({msg})")
            return False
        moves = len(sched)
        methods.append((name, moves, wall, ""))
        log(f"  {name}: {moves} steps in {wall:.1f}s")
        if best is None or moves < best[0]:
            best = (moves, order, sched, name)
        return True

    # --- stage 1: LCS route (always) -------------------------------------
    t0 = time.time()
    try:
        order, sched, info = method_lcs_route(ids, seqs_by_id,
                                              cfg["lcs_s"], cfg["hk_cutoff"])
        consider("LCS route", order, sched, time.time() - t0)
        if info["hk"]:
            hk_lb = max(max_len, baseline - info["path_weight"])
            if hk_lb > lb:
                lb = hk_lb
            lb_exactness = "exact ordering bound"
    except Exception as e:
        methods.append(("LCS route", None, time.time() - t0, f"error: {e}"))
        log(f"  LCS route failed: {e}")
    if best and best[0] <= lb:
        proven = f"matches the {lb_exactness} ({lb} steps)"

    # --- stage 2: exact search (when plausible within cap) ---------------
    if proven is None and cfg["exact_cap"] > 0 and np is not None \
            and n <= _EXACT_MEM_LIMIT_N \
            and (_predict_exact_seconds(n) <= cfg["exact_cap"] * 1.5
                 or n <= 12):
        t0 = time.time()
        log(f"  exact search (cap {cfg['exact_cap']}s)...")
        try:
            res = method_exact(ids, seqs_by_id, cfg["exact_cap"])
        except Exception as e:
            res = None
            log(f"  exact search failed: {e}")
        wall = time.time() - t0
        if res is None:
            methods.append(("exact search", None, wall,
                            "stopped at time cap (no proof)"))
            log(f"  exact search: stopped at {wall:.0f}s cap")
        else:
            order, sched, _ = res
            if consider("exact search", order, sched, wall):
                proven = "exact search completed"
                lb = min(best[0], len(sched))

    # --- stage 3: simulated annealing -------------------------------------
    if proven is None and cfg["sa_s"] > 0:
        t0 = time.time()
        try:
            order, sched, _ = method_anneal(ids, seqs_by_id, cfg["sa_s"])
            consider("annealing", order, sched, time.time() - t0)
        except Exception as e:
            methods.append(("annealing", None, time.time() - t0,
                            f"error: {e}"))
            log(f"  annealing failed: {e}")
        if best and best[0] <= lb:
            proven = f"matches the {lb_exactness} ({lb} steps)"

    # --- stage 4: beam search ---------------------------------------------
    if proven is None and cfg["beam_s"] > 0:
        t0 = time.time()
        try:
            order, sched, _ = method_beam(ids, seqs_by_id, cfg["beam_s"],
                                          cfg["bw"])
            consider("beam search", order, sched, time.time() - t0)
        except Exception as e:
            methods.append(("beam search", None, time.time() - t0,
                            f"error: {e}"))
            log(f"  beam search failed: {e}")
        if best and best[0] <= lb:
            proven = f"matches the {lb_exactness} ({lb} steps)"

    if best is None:
        raise RuntimeError("no method produced a valid schedule")

    moves, order, sched, method = best
    return dict(order=order, schedule=sched, moves=moves, baseline=baseline,
                status="PROVEN OPTIMAL" if proven else "BEST FOUND",
                proof=proven or f"within {moves - lb} step(s) of the "
                                f"lower bound ({lb})",
                best_method=method, methods=methods, lower_bound=lb,
                wall_s=time.time() - t_start)


# ============================================================================
# Report formatting (slots shown 1-based for humans)
# ============================================================================

def format_report(seqs_by_id, result):
    order, sched = result["order"], result["schedule"]
    moves, baseline = result["moves"], result["baseline"]
    lines = []
    ap = lines.append
    ap("=" * 66)
    ap("DEPOSITION PLAN")
    ap("=" * 66)
    ap(f"Strips:            {len(order)}")
    ap(f"Total steps:       {moves}   (serial baseline {baseline})")
    ap(f"Speedup:           {baseline / moves:.2f}x   "
       f"(efficiency {100 * (1 - moves / baseline):.1f}%)")
    swaps = sum(1 for t in range(1, len(sched))
                if sched[t][0] != sched[t - 1][0])
    ap(f"Target changes:    {swaps}")
    ap(f"Optimality:        {result['status']} — {result['proof']}")
    ap(f"Compute time:      {result['wall_s']:.1f} s")
    ap("")
    ap("LOAD ORDER (slot 1 = left edge)")
    for k, g in enumerate(order):
        ap(f"  slot {k + 1:2d}:  strip {g:>3}   {'-'.join(seqs_by_id[g])}")
    ap("")
    ap("SCHEDULE  (window = slots exposed, inclusive, 1-based)")
    for t, (e, i, j) in enumerate(sched):
        strips = ", ".join(str(order[k]) for k in range(i, j + 1))
        ap(f"  step {t + 1:3d}:  {e:<3} window slots {i + 1:2d}-{j + 1:2d}"
           f"   strips {strips}")
    ap("")
    ap("METHODS TRIED")
    for name, m, wall, note in result["methods"]:
        mv = f"{m} steps" if m is not None else "-"
        note = f"  ({note})" if note else ""
        ap(f"  {name:<14} {mv:>10}   {wall:6.1f} s{note}")
    ap("=" * 66)
    return "\n".join(lines) + "\n"


# ============================================================================
# Step-by-step state frames (used by the GUI viewer and saved plan files)
# ============================================================================

def _ptr_snapshots(seqs, schedule):
    """Per-strip progress after each step; snaps[0] = initial state."""
    n = len(seqs)
    snaps = [[0] * n]
    cur = [0] * n
    for (_, i, j) in schedule:
        cur = cur[:]
        for k in range(i, j + 1):
            cur[k] += 1
        snaps.append(cur)
    return snaps


def render_frame(order, seqs, schedule, snaps, t):
    """One frame = state after step t (t = 0 is the empty initial state).

    Returns (header, rows); each row is
    (prefix, [(cell_text, tag), ...], suffix, exposed) with tags
    'done' / 'new' / 'dim' so the GUI can colour cells; joining the cell
    texts gives the aligned plain-text version.
    """
    n = len(seqs)
    w = max(len(e) for s in seqs for e in s)
    total = sum(len(s) for s in seqs)
    done_total = sum(snaps[t])
    if t == 0:
        header = (f"Initial state — strips loaded, nothing deposited "
                  f"(0/{total} layers)")
        wi = wj = -1
    else:
        e, wi, wj = schedule[t - 1]
        strips = ", ".join(str(order[k]) for k in range(wi, wj + 1))
        header = (f"Step {t} of {len(schedule)}:  deposit {e}  —  mask window "
                  f"slots {wi + 1}-{wj + 1}  (strips {strips})    "
                  f"{done_total}/{total} layers done")
    rows = []
    for k in range(n):
        exposed = wi <= k <= wj
        p = snaps[t][k]
        prefix = ("▶" if exposed else " ") + \
            f" slot {k + 1:2d}  strip {order[k]:>3}  "
        segs = []
        for li, elem in enumerate(seqs[k]):
            if exposed and li == p - 1:
                segs.append((f"[{elem:>{w}}]", "new"))
            elif li < p:
                segs.append((f" {elem:>{w}} ", "done"))
            else:
                segs.append((f" {'·':>{w}} ", "dim"))
        suffix = f"  {p:2d}/{len(seqs[k])}"
        rows.append((prefix, segs, suffix, exposed))
    return header, rows


def render_frames_text(seqs_by_id, order, schedule):
    """Full plain-text step-by-step dump for saved plan files."""
    seqs = [seqs_by_id[g] for g in order]
    snaps = _ptr_snapshots(seqs, schedule)
    out = ["=" * 66,
           "STEP-BY-STEP VIEW",
           "[El] = deposited this step   · = not yet deposited   "
           "▶ = slot exposed",
           "=" * 66]
    for t in range(len(schedule) + 1):
        header, rows = render_frame(order, seqs, schedule, snaps, t)
        out.append("")
        out.append(header)
        for prefix, segs, suffix, _ in rows:
            out.append(prefix + "".join(s for s, _ in segs) + suffix)
    return "\n".join(out) + "\n"


# ============================================================================
# CLI
# ============================================================================

def run_cli(text, effort, out_path):
    seqs = parse_strips(text)
    print(f"Parsed {len(seqs)} strips.")
    result = plan(seqs, effort, log=print)
    report = format_report(seqs, result)
    print()
    print(report)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n")
            f.write(render_frames_text(seqs, result["order"],
                                       result["schedule"]))
        print(f"Saved to {out_path} (includes the step-by-step view)")


# ============================================================================
# GUI (tkinter)
# ============================================================================

_EFFORT_HINTS = {
    1: "1 — quick look (~seconds, no proof)",
    2: "2 — fast (~10 s, proof for small sets)",
    3: "3 — standard (~30-60 s, proof up to ~20 strips)  [default]",
    4: "4 — thorough (~2-3 min)",
    5: "5 — maximum (~10-15 min, best possible)",
}

_PLACEHOLDER = """\
# Paste your strips here, then press Run.
# Accepted formats:
#   test_genomes = { 1: 'Mn-Co-Al-Co-...', 2: 'Mn-Fe-...', }
#   1: Mn-Co-Al-Co
#   Mn Fe Al Co
"""


_COL = {
    "bg":     "#f4f1ec",   # window background
    "panel":  "#ffffff",   # text panels
    "ink":    "#2b2a27",   # main text
    "accent": "#b3541e",   # run button / highlights (warm copper)
    "accent2": "#1d5c8f",  # exposed-slot blue
    "done":   "#2e7d43",   # deposited layers
    "new":    "#c2410c",   # layers deposited this step
    "dim":    "#b5b0a8",   # pending layers
    "newbg":  "#fdeee2",   # background of freshly deposited cells
}


def run_gui(effort_default=3):
    import queue as _queue
    import threading
    import tkinter as tk
    from tkinter import filedialog, font, ttk

    root = tk.Tk()
    root.title("Epitaxy Deposition Planner")
    root.geometry("1320x920")
    root.configure(bg=_COL["bg"])

    # ---- fonts: readable sizes everywhere -------------------------------
    for fname in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
                  "TkHeadingFont"):
        font.nametofont(fname).configure(size=14)
    mono = font.nametofont("TkFixedFont")
    mono.configure(size=15)
    mono_big = mono.copy()
    mono_big.configure(size=15, weight="bold")
    h1 = font.nametofont("TkDefaultFont").copy()
    h1.configure(size=20, weight="bold")
    h2 = font.nametofont("TkDefaultFont").copy()
    h2.configure(size=14, weight="bold")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=_COL["bg"], foreground=_COL["ink"],
                    font=("TkDefaultFont", 14))
    style.configure("TNotebook", background=_COL["bg"], borderwidth=0)
    style.configure("TNotebook.Tab", padding=(20, 10),
                    font=("TkDefaultFont", 14, "bold"))
    style.map("TNotebook.Tab",
              background=[("selected", _COL["panel"])],
              foreground=[("selected", _COL["accent"])])
    style.configure("Run.TButton", font=("TkDefaultFont", 15, "bold"),
                    padding=(24, 11), background=_COL["accent"],
                    foreground="white")
    style.map("Run.TButton",
              background=[("active", "#96431a"), ("disabled", "#cbb9a8")])
    style.configure("TButton", padding=(12, 7))
    style.configure("TScale", background=_COL["bg"])

    def make_text(parent, **kw):
        """Read-only-styled text panel with a scrollbar."""
        frame = ttk.Frame(parent)
        txt = tk.Text(frame, font=mono, bg=_COL["panel"], fg=_COL["ink"],
                      relief="flat", padx=14, pady=12, wrap="none",
                      insertbackground=_COL["ink"], **kw)
        sy = ttk.Scrollbar(frame, command=txt.yview)
        sx = ttk.Scrollbar(frame, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        sy.pack(side="right", fill="y")
        sx.pack(side="bottom", fill="x")
        txt.pack(side="left", fill="both", expand=True)
        return frame, txt

    # ---- header ----------------------------------------------------------
    head = tk.Frame(root, bg=_COL["bg"])
    head.pack(fill="x", padx=18, pady=(14, 4))
    tk.Label(head, text="Epitaxy Deposition Planner", font=h1,
             bg=_COL["bg"], fg=_COL["ink"]).pack(side="left")
    status = tk.Label(head, text="", font=h2, bg=_COL["bg"],
                      fg=_COL["accent2"])
    status.pack(side="right")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=18, pady=(6, 14))

    # ---- tab 1: input ------------------------------------------------------
    tab1 = ttk.Frame(nb)
    nb.add(tab1, text="  1 · Strips  ")
    tk.Label(tab1, text="Paste your strip recipes:", font=h2,
             bg=_COL["bg"], fg=_COL["ink"], anchor="w"
             ).pack(fill="x", pady=(12, 4))
    inp_frame, inp = make_text(tab1, height=16)
    inp.configure(state="normal", wrap="word")
    inp_frame.pack(fill="both", expand=True)
    inp.insert("1.0", _PLACEHOLDER)

    dial = tk.Frame(tab1, bg=_COL["bg"])
    dial.pack(fill="x", pady=12)
    tk.Label(dial, text="Effort", font=h2, bg=_COL["bg"],
             fg=_COL["ink"]).pack(side="left")
    hint = tk.Label(dial, text=_EFFORT_HINTS[effort_default],
                    bg=_COL["bg"], fg=_COL["accent2"],
                    font=("TkDefaultFont", 13))
    scale = ttk.Scale(dial, from_=1, to=5, orient="horizontal", length=260,
                      command=lambda v: (hint.config(
                          text=_EFFORT_HINTS[int(float(v) + 0.5)])))
    scale.set(effort_default)
    scale.pack(side="left", padx=14)
    hint.pack(side="left", padx=6)
    run_btn = ttk.Button(dial, text="▶   Run", style="Run.TButton")
    run_btn.pack(side="right", padx=4)

    # ---- tab 2: plan -------------------------------------------------------
    tab2 = ttk.Frame(nb)
    nb.add(tab2, text="  2 · Plan  ")
    bar2 = tk.Frame(tab2, bg=_COL["bg"])
    bar2.pack(fill="x", pady=(10, 4))
    save_btn = ttk.Button(bar2, text="Save plan…", state="disabled")
    save_btn.pack(side="left")
    plan_frame, plan_txt = make_text(tab2)
    plan_frame.pack(fill="both", expand=True, pady=(4, 8))
    plan_txt.configure(state="disabled")
    plan_txt.tag_configure("hdr", font=mono_big, foreground=_COL["accent"])

    # ---- tab 3: step-by-step viewer -----------------------------------------
    tab3 = ttk.Frame(nb)
    nb.add(tab3, text="  3 · Step-by-step  ")
    bar3 = tk.Frame(tab3, bg=_COL["bg"])
    bar3.pack(fill="x", pady=(10, 4))
    prev_btn = ttk.Button(bar3, text="◀  Prev", state="disabled")
    prev_btn.pack(side="left")
    next_btn = ttk.Button(bar3, text="Next  ▶", state="disabled")
    next_btn.pack(side="left", padx=(8, 16))
    step_var = tk.IntVar(value=0)
    step_scale = ttk.Scale(bar3, from_=0, to=1, orient="horizontal",
                           length=420, state="disabled")
    step_scale.pack(side="left", padx=4, fill="x", expand=True)
    step_lbl = tk.Label(bar3, text="run a plan first", font=h2,
                        bg=_COL["bg"], fg=_COL["ink"])
    step_lbl.pack(side="right", padx=8)

    view_frame, view = make_text(tab3)
    view_frame.pack(fill="both", expand=True, pady=(4, 2))
    view.configure(state="disabled")
    view.tag_configure("head", font=mono_big, foreground=_COL["ink"])
    view.tag_configure("done", foreground=_COL["done"])
    view.tag_configure("new", foreground=_COL["new"], font=mono_big,
                       background=_COL["newbg"])
    view.tag_configure("dim", foreground=_COL["dim"])
    view.tag_configure("exp", foreground=_COL["accent2"], font=mono_big)
    view.tag_configure("plain", foreground=_COL["ink"])
    tk.Label(tab3, text="highlighted = deposited this step     · = not yet "
             "deposited     ▶ = slot inside the mask window     "
             "(←/→ arrow keys work too)",
             bg=_COL["bg"], fg=_COL["dim"], font=("TkDefaultFont", 12)
             ).pack(fill="x", pady=(0, 8))

    # ---- state + worker ------------------------------------------------------
    q = _queue.Queue()
    state = {"running": False, "seqs": None, "result": None,
             "snaps": None, "slot_seqs": None}

    def emit(msg):
        q.put(("log", msg))

    def worker(text, effort):
        try:
            seqs = parse_strips(text)
            emit(f"Parsed {len(seqs)} strips.")
            result = plan(seqs, effort, log=emit)
            q.put(("done", (seqs, result)))
        except Exception as e:
            q.put(("error", f"ERROR: {e}"))

    def plan_append(msg, tag="plain"):
        plan_txt.config(state="normal")
        plan_txt.insert("end", msg + "\n", tag)
        plan_txt.see("end")
        plan_txt.config(state="disabled")

    # ---- step viewer -----------------------------------------------------------
    def show_step(t):
        if state["result"] is None:
            return
        sched = state["result"]["schedule"]
        t = max(0, min(len(sched), int(t)))
        step_var.set(t)
        order = state["result"]["order"]
        header, rows = render_frame(order, state["slot_seqs"], sched,
                                    state["snaps"], t)
        view.config(state="normal")
        view.delete("1.0", "end")
        view.insert("end", header + "\n\n", "head")
        for prefix, segs, suffix, exposed in rows:
            view.insert("end", prefix, "exp" if exposed else "plain")
            for s, tag in segs:
                view.insert("end", s, tag)
            view.insert("end", suffix + "\n", "dim")
        view.config(state="disabled")
        step_lbl.config(text=f"step {t} / {len(sched)}")

    def on_scale(v):
        t = int(float(v) + 0.5)
        if t != step_var.get():
            show_step(t)

    def nudge(d):
        show_step(step_var.get() + d)

    step_scale.configure(command=on_scale)
    prev_btn.configure(command=lambda: nudge(-1))
    next_btn.configure(command=lambda: nudge(+1))
    root.bind("<Left>", lambda e: nudge(-1) if state["result"] else None)
    root.bind("<Right>", lambda e: nudge(+1) if state["result"] else None)

    # ---- run / done / save --------------------------------------------------
    def finish(seqs, result):
        state.update(seqs=seqs, result=result, running=False)
        order, sched = result["order"], result["schedule"]
        state["slot_seqs"] = [seqs[g] for g in order]
        state["snaps"] = _ptr_snapshots(state["slot_seqs"], sched)
        plan_append("")
        for line in format_report(seqs, result).splitlines():
            tag = "hdr" if (set(line) == {"="} or line in
                            ("DEPOSITION PLAN",)) else "plain"
            plan_append(line, tag)
        run_btn.config(state="normal")
        save_btn.config(state="normal")
        prev_btn.config(state="normal")
        next_btn.config(state="normal")
        step_scale.configure(state="normal", from_=0, to=len(sched))
        status.config(text=f"done — {result['moves']} steps, "
                           f"{result['status'].lower()}")
        show_step(0)
        nb.select(tab2)

    def poll():
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "log":
                    plan_append(payload)
                elif kind == "done":
                    finish(*payload)
                else:
                    plan_append(payload)
                    state["running"] = False
                    run_btn.config(state="normal")
                    status.config(text="failed")
        except _queue.Empty:
            pass
        root.after(120, poll)

    def on_run():
        if state["running"]:
            return
        state.update(running=True, result=None)
        run_btn.config(state="disabled")
        save_btn.config(state="disabled")
        status.config(text="running…")
        plan_txt.config(state="normal")
        plan_txt.delete("1.0", "end")
        plan_txt.config(state="disabled")
        nb.select(tab2)
        threading.Thread(
            target=worker,
            args=(inp.get("1.0", "end"), int(float(scale.get()) + 0.5)),
            daemon=True).start()

    def on_save():
        if state["result"] is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", initialfile="deposition_plan.txt")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(format_report(state["seqs"], state["result"]))
                f.write("\n")
                f.write(render_frames_text(state["seqs"],
                                           state["result"]["order"],
                                           state["result"]["schedule"]))
            status.config(text=f"saved: {os.path.basename(path)}")

    run_btn.config(command=on_run)
    save_btn.config(command=on_save)
    root.after(120, poll)
    root.mainloop()


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Epitaxy deposition planner: paste strips, get an "
                    "optimal(ish) arrangement + schedule.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Effort: 1 quick / 2 fast / 3 standard (default) / "
               "4 thorough / 5 maximum")
    ap.add_argument("input", nargs="?",
                    help="file with strip recipes (omit to use GUI or stdin)")
    ap.add_argument("-e", "--effort", type=int, default=3,
                    choices=[1, 2, 3, 4, 5], help="optimality dial, 1-5 "
                    "(default 3)")
    ap.add_argument("-o", "--out", help="also save the plan to this file")
    ap.add_argument("--cli", action="store_true",
                    help="force terminal mode (no GUI)")
    args = ap.parse_args()

    if args.input:
        with open(args.input) as f:
            text = f.read()
        run_cli(text, args.effort, args.out)
        return
    if not sys.stdin.isatty():
        run_cli(sys.stdin.read(), args.effort, args.out)
        return
    if not args.cli:
        try:
            run_gui(args.effort)
            return
        except Exception as e:
            print(f"(GUI unavailable: {e} — falling back to paste mode)\n")
    print("Paste your strip recipes below, then press Ctrl-D on an "
          "empty line:")
    run_cli(sys.stdin.read(), args.effort, args.out)


if __name__ == "__main__":
    main()
