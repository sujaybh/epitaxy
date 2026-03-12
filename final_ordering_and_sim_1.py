import time
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional


# =============================================================================
# OPTIMIZED DEPOSITION SIMULATOR (from Script 1)
# =============================================================================

class DepositionSimulator:
    """
    OPTIMIZED SIMULATOR (Greedy Segments + Fast Hashing)
    Maintains original logging and statistics format.
    """

    def __init__(self, genomes: Dict[int, str], genome_order: List[int],
                 pattern_length: int = 1, beam_width: int = 3):
        self.base_genomes = genomes
        self.genome_order = genome_order
        self.num_strips = len(genome_order)
        self.pattern_length = pattern_length
        self.beam_width = beam_width

        # Parse and expand patterns
        self.genomes = {}
        for genome_id, pattern in genomes.items():
            elements = self._parse_sequence(pattern)
            self.genomes[genome_id] = elements * pattern_length

        # Store target sequences
        self.strip_genomes = [self.genomes[gid] for gid in self.genome_order]
        self.strip_lengths = [len(seq) for seq in self.strip_genomes]

        # OPTIMIZATION: Pre-compute Element Map for O(1) lookup
        self.element_map = []
        for seq in self.strip_genomes:
            self.element_map.append({i: e for i, e in enumerate(seq)})

        # Statistics
        self.nodes_explored = 0
        self.max_depth_reached = 0
        self.moves_pruned = 0
        self.bounds_pruned = 0
        self.cache_hits = 0
        self.state_cache = {}

    def _parse_sequence(self, sequence_string: str) -> List[str]:
        if '-' in sequence_string:
            return [e.strip() for e in sequence_string.split('-') if e.strip()]
        return list(sequence_string)

    def _sequence_to_string(self, sequence: List[str]) -> str:
        if len(sequence) > 0 and len(sequence[0]) > 1:
            return '-'.join(sequence)
        return ''.join(sequence)

    def _reconstruct_state_for_logging(self, strip_progress: Tuple[int, ...]) -> List[str]:
        state_strs = []
        for i in range(self.num_strips):
            current_seq = self.strip_genomes[i][:strip_progress[i]]
            state_str = self._sequence_to_string(current_seq)
            state_strs.append(f"Strip {i}: {state_str}")
        return state_strs

    def _calculate_lower_bound(self, strip_progress: Tuple[int, ...], current_moves: int) -> int:
        max_rem = 0
        frontier_elements = set()

        for i in range(self.num_strips):
            rem = self.strip_lengths[i] - strip_progress[i]
            if rem > 0:
                if rem > max_rem:
                    max_rem = rem
                frontier_elements.add(self.element_map[i][strip_progress[i]])

        conflict_penalty = max(0, len(frontier_elements) - 1)
        return current_moves + max_rem + conflict_penalty

    def get_smart_moves(self, strip_progress: Tuple[int, ...]) -> List[Tuple[str, Tuple[int, ...]]]:
        """
        OPTIMIZATION: Greedy Maximal Segments.
        Returns only the largest contiguous block for each active element.
        """
        moves = []

        active_needs = []
        for i in range(self.num_strips):
            if strip_progress[i] < self.strip_lengths[i]:
                el = self.element_map[i][strip_progress[i]]
                active_needs.append((i, el))
            else:
                active_needs.append((i, None))

        i = 0
        while i < self.num_strips:
            current_el = active_needs[i][1]
            if current_el is None:
                i += 1
                continue

            segment_start = i
            while i < self.num_strips and active_needs[i][1] == current_el:
                i += 1

            mask = tuple(range(segment_start, i))
            moves.append((current_el, mask))

        return moves

    def score_move(self, element: str, mask: Tuple[int, ...], strip_progress: Tuple[int, ...]) -> float:
        progress = len(mask)

        lookahead_bonus = 0
        next_counts = {}
        for idx in mask:
            nxt = strip_progress[idx] + 1
            if nxt < self.strip_lengths[idx]:
                el = self.element_map[idx][nxt]
                next_counts[el] = next_counts.get(el, 0) + 1
        if next_counts:
            lookahead_bonus = max(next_counts.values()) * 0.5

        urgency = 0
        for idx in mask:
            rem = self.strip_lengths[idx] - strip_progress[idx]
            urgency += rem * 0.1

        return (progress * 5.0) + lookahead_bonus + urgency

    def _dfs_search(self, strip_progress: Tuple[int, ...], current_moves: int,
                    current_log: List[str], depth: int,
                    best_known: Optional[int] = None) -> Tuple[Optional[int], Optional[List[str]]]:

        self.nodes_explored += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)

        lower_bound = self._calculate_lower_bound(strip_progress, current_moves)
        if best_known is not None and lower_bound >= best_known:
            self.bounds_pruned += 1
            return (None, None)

        if strip_progress in self.state_cache:
            rem = self.state_cache[strip_progress]
            if best_known is not None and (current_moves + rem >= best_known):
                self.cache_hits += 1
                self.bounds_pruned += 1
                return (None, None)

        is_complete = all(strip_progress[i] >= self.strip_lengths[i] for i in range(self.num_strips))
        if is_complete:
            self.state_cache[strip_progress] = 0
            return (current_moves, current_log)

        valid_moves = self.get_smart_moves(strip_progress)

        scored_moves = []
        for el, mask in valid_moves:
            score = self.score_move(el, mask, strip_progress)
            scored_moves.append((score, el, mask))

        scored_moves.sort(key=lambda x: x[0], reverse=True)

        beam_limit = min(self.beam_width, len(scored_moves))
        top_moves = scored_moves[:beam_limit]

        if len(scored_moves) > beam_limit:
            self.moves_pruned += (len(scored_moves) - beam_limit)

        best_total_moves = best_known
        best_log = None
        min_rem_found = float('inf')

        for _, element, mask_config in top_moves:
            new_progress_list = list(strip_progress)
            useful_strips = []

            for idx in mask_config:
                new_progress_list[idx] += 1
                useful_strips.append(idx)

            new_strip_progress = tuple(new_progress_list)

            new_log = list(current_log)
            move_num = current_moves + 1
            mask_list = list(mask_config)
            new_log.append(f"Move {move_num}: Deposit {element} on strips {mask_list} (needed by: {useful_strips})")

            state_strs = self._reconstruct_state_for_logging(new_strip_progress)
            new_log.append(f"  State: {' | '.join(state_strs)}")

            result_moves, result_log = self._dfs_search(
                new_strip_progress,
                current_moves + 1,
                new_log,
                depth + 1,
                best_total_moves
            )

            if result_moves is not None:
                if best_total_moves is None or result_moves < best_total_moves:
                    best_total_moves = result_moves
                    best_log = result_log

                rem = result_moves - current_moves
                min_rem_found = min(min_rem_found, rem)

        if min_rem_found != float('inf'):
            self.state_cache[strip_progress] = min_rem_found

        return (best_total_moves, best_log)

    def simulate(self, verbose: bool = True) -> Tuple[int, List[str], Dict]:
        self.nodes_explored = 0
        self.max_depth_reached = 0
        self.moves_pruned = 0
        self.bounds_pruned = 0
        self.cache_hits = 0
        self.state_cache = {}

        strip_progress = tuple([0] * self.num_strips)

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"RUNNING OPTIMIZED SIMULATION ({self.num_strips} STRIPS, BEAM WIDTH {self.beam_width})")
            print(f"{'=' * 80}\n")
            print(f"Genome Assignment:")
            for strip_idx, genome_id in enumerate(self.genome_order):
                seq = self._sequence_to_string(self.strip_genomes[strip_idx])
                print(f"  Strip {strip_idx} ← Genome {genome_id}: {seq}")
            print()

        start_time = time.time()
        min_moves, operation_log = self._dfs_search(strip_progress, 0, [], 0, None)
        end_time = time.time()

        if min_moves is None:
            return (float('inf'), ["Error: No valid deposition sequence found"], {})

        stats = {
            'execution_time': end_time - start_time,
            'total_moves': min_moves,
            'nodes_explored': self.nodes_explored,
            'moves_pruned': self.moves_pruned,
            'bounds_pruned': self.bounds_pruned,
            'cache_hits': self.cache_hits,
            'max_depth_reached': self.max_depth_reached,
            'beam_width': self.beam_width,
        }

        return (min_moves, operation_log, stats)


# =============================================================================
# WORKER FUNCTION FOR PARALLEL POSITION EVALUATION
# =============================================================================

def evaluate_position_worker(args):
    """Worker for parallel position evaluation using the optimized simulator."""
    genomes_dict, pattern_length, pos, current_order, genome_to_insert = args
    trial_order = current_order[:pos] + [genome_to_insert] + current_order[pos:]

    sim = DepositionSimulator(
        genomes=genomes_dict,
        genome_order=trial_order,
        pattern_length=pattern_length,
        beam_width=2  # Fast ordering phase uses beam_width=2
    )
    total_moves, _, _ = sim.simulate(verbose=False)
    return (pos, total_moves, trial_order)


# =============================================================================
# INSERTION ORDERING + FINAL HIGH-QUALITY SIMULATION
# =============================================================================

class EpitaxyOrderingAndSimulator:
    """
    Combines constructive insertion ordering (beam_width=2 for speed)
    with a final high-quality simulation (beam_width=3) on the best order found.
    """

    def __init__(self, genomes: Dict[int, str], genome_order: List[int],
                 pattern_length: int = 1, max_workers: Optional[int] = None):
        self.genomes = genomes
        self.genome_order = genome_order  # The specific genome IDs to use (subset of dict)
        self.pattern_length = pattern_length
        self.max_workers = max_workers or mp.cpu_count()

        print("=" * 80)
        print("CRYSTAL GENOME EPITAXY DEPOSITION SCHEDULER (OPTIMIZED + ORDERING)")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Genomes to schedule: {genome_order}")
        print(f"  Number of strips: {len(genome_order)}")
        print(f"  Pattern length: {pattern_length}x")
        print(f"  Ordering phase beam width: 2 (fast)")
        print(f"  Final simulation beam width: 3 (high quality)")
        print(f"  Parallel workers: {self.max_workers}")
        print(f"\nGenome Dictionary:")
        for gid in genome_order:
            print(f"  Genome {gid}: {genomes[gid]}")

    def constructive_insertion_ordering(self) -> Tuple[List[int], int, List[str]]:
        """
        Build optimal strip ordering by inserting one genome at a time.
        Returns (optimal_order, best_moves_at_bw2, best_log_at_bw2).
        """
        ids = list(self.genome_order)

        if len(ids) == 1:
            sim = DepositionSimulator(self.genomes, ids, self.pattern_length, beam_width=2)
            moves, log, _ = sim.simulate(verbose=False)
            return ids, moves, log

        current_order = [ids[0], ids[1]]
        print(f"\n{'=' * 80}")
        print(f"PHASE 1: CONSTRUCTIVE INSERTION ORDERING (beam_width=2)")
        print(f"{'=' * 80}")
        print(f"  Starting with: Genomes {current_order}")

        last_best_moves = None
        last_best_log = None

        for i in range(2, len(ids)):
            genome_to_insert = ids[i]
            positions = list(range(len(current_order) + 1))

            print(f"\n  Inserting Genome {genome_to_insert} — evaluating {len(positions)} positions in parallel...")

            worker_args = [
                (self.genomes, self.pattern_length, pos, current_order, genome_to_insert)
                for pos in positions
            ]

            with mp.Pool(processes=min(len(positions), self.max_workers)) as pool:
                results = pool.map(evaluate_position_worker, worker_args)

            best_pos, best_moves, best_order = None, float('inf'), None
            for pos, moves, trial_order in sorted(results, key=lambda x: x[0]):
                marker = " ← BEST" if moves < best_moves else ""
                print(f"    Position {pos}: {moves} moves{marker}")
                if moves < best_moves:
                    best_moves = moves
                    best_pos = pos
                    best_order = trial_order

            current_order = best_order
            last_best_moves = best_moves
            print(f"  → Selected position {best_pos} → Order: {current_order}")

        print(f"\n  Final ordering determined: {current_order}")

        # Run once more on final order to capture full log
        sim = DepositionSimulator(self.genomes, current_order, self.pattern_length, beam_width=2)
        last_best_moves, last_best_log, _ = sim.simulate(verbose=False)

        return current_order, last_best_moves, last_best_log

    def run(self, run_final_beam3: bool = True) -> Tuple[int, List[str], Dict]:
        """
        Run full pipeline.

        Args:
            run_final_beam3: If True, re-run final simulation at beam_width=3 after ordering.
                             If False, use the beam_width=2 result from the last ordering step.
        """

        # --- Phase 1: Find optimal ordering ---
        ordering_start = time.time()
        optimal_order, best_moves_bw2, best_log_bw2 = self.constructive_insertion_ordering()
        ordering_time = time.time() - ordering_start

        print(f"\n  Ordering phase complete in {ordering_time:.3f}s")
        print(f"  Optimal order: {optimal_order}")
        print(f"  Best result (beam_width=2): {best_moves_bw2} moves")

        if not run_final_beam3:
            # Use the beam_width=2 result directly — re-run once on optimal order for full log
            print(f"\n{'=' * 80}")
            print(f"PHASE 2: SKIPPED — Using beam_width=2 result")
            print(f"{'=' * 80}")

            # Re-run on optimal order at bw=2 to get fresh stats + full log
            final_sim = DepositionSimulator(
                genomes=self.genomes,
                genome_order=optimal_order,
                pattern_length=self.pattern_length,
                beam_width=2
            )
            total_moves, operation_log, stats = final_sim.simulate(verbose=True)
            stats['ordering_time'] = ordering_time
            stats['optimal_order'] = optimal_order
            stats['final_beam_width'] = 2
            return total_moves, operation_log, stats

        # --- Phase 2: Final simulation with beam_width=3 ---
        print(f"\n{'=' * 80}")
        print(f"PHASE 2: FINAL HIGH-QUALITY SIMULATION (beam_width=3)")
        print(f"{'=' * 80}")

        final_sim = DepositionSimulator(
            genomes=self.genomes,
            genome_order=optimal_order,
            pattern_length=self.pattern_length,
            beam_width=3
        )
        total_moves, operation_log, stats = final_sim.simulate(verbose=True)

        stats['ordering_time'] = ordering_time
        stats['optimal_order'] = optimal_order
        stats['final_beam_width'] = 3

        return total_moves, operation_log, stats


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_detailed_moves(operation_log: List[str], min_moves: int):
    print(f"\n{'=' * 80}")
    print(f"DETAILED DEPOSITION INSTRUCTIONS ({min_moves} total moves)")
    print(f"{'=' * 80}\n")

    move_count = 0
    for line in operation_log:
        if line.startswith("Move"):
            move_count += 1
            parts = line.split(":")
            move_info = parts[1].strip()
            print(f"\n{'─' * 80}")
            print(f"STEP {move_count}:")
            print(f"  ACTION: {move_info}")
        elif line.startswith("  State:"):
            state_info = line.replace("  State: ", "")
            strips = state_info.split(" | ")
            print(f"  RESULT:")
            for strip in strips:
                print(f"    {strip}")

    print(f"\n{'─' * 80}")
    print(f"✓ All genomes completed in {min_moves} deposition moves!")
    print(f"{'=' * 80}\n")


def build_output_text(
    genome_order: List[int],
    test_genomes: Dict[int, str],
    optimal_order: List[int],
    total_moves: int,
    operation_log: List[str],
    stats: Dict,
    pattern_length: int
) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("CRYSTAL GENOME EPITAXY DEPOSITION SCHEDULER — RESULTS")
    lines.append("=" * 80)
    lines.append(f"")
    lines.append(f"Configuration:")
    lines.append(f"  Requested genomes : {genome_order}")
    lines.append(f"  Pattern length    : {pattern_length}x")
    lines.append(f"  Ordering beam     : 2")
    lines.append(f"  Final beam        : {stats.get('final_beam_width', '?')}")
    lines.append(f"")
    lines.append(f"Genome Dictionary:")
    for gid in genome_order:
        lines.append(f"  Genome {gid}: {test_genomes[gid]}")
    lines.append(f"")
    lines.append("=" * 80)
    lines.append("OPTIMAL STRIP ORDERING")
    lines.append("=" * 80)
    lines.append(f"  Order : {optimal_order}")
    for strip_idx, gid in enumerate(optimal_order):
        lines.append(f"  Strip {strip_idx} ← Genome {gid}: {test_genomes[gid]}")
    lines.append(f"")
    lines.append("=" * 80)
    lines.append(f"FINAL SIMULATION RESULTS (beam_width=3)")
    lines.append("=" * 80)
    lines.append(f"  Total Deposition Moves : {total_moves}")
    lines.append(f"  Ordering Phase Time    : {stats.get('ordering_time', 0):.3f}s")
    lines.append(f"  Final Sim Time         : {stats.get('execution_time', 0):.3f}s")
    lines.append(f"")
    lines.append(f"Search Statistics:")
    lines.append(f"  Nodes explored  : {stats.get('nodes_explored', 0):,}")
    lines.append(f"  Moves pruned    : {stats.get('moves_pruned', 0):,}")
    lines.append(f"  Bounds pruned   : {stats.get('bounds_pruned', 0):,}")
    lines.append(f"  Cache hits      : {stats.get('cache_hits', 0):,}")
    lines.append(f"  Max depth       : {stats.get('max_depth_reached', 0)}")
    lines.append(f"")
    lines.append("=" * 80)
    lines.append(f"DETAILED DEPOSITION INSTRUCTIONS ({total_moves} total moves)")
    lines.append("=" * 80)

    move_count = 0
    for line in operation_log:
        if line.startswith("Move"):
            move_count += 1
            parts = line.split(":")
            move_info = parts[1].strip()
            lines.append(f"")
            lines.append("─" * 80)
            lines.append(f"STEP {move_count}:")
            lines.append(f"  ACTION: {move_info}")
        elif line.startswith("  State:"):
            state_info = line.replace("  State: ", "")
            strips = state_info.split(" | ")
            lines.append(f"  RESULT:")
            for strip in strips:
                lines.append(f"    {strip}")

    lines.append(f"")
    lines.append("─" * 80)
    lines.append(f"✓ All genomes completed in {total_moves} deposition moves!")
    lines.append("")

    # Efficiency
    genome_length = len([e for e in test_genomes[genome_order[0]].split('-') if e]) * pattern_length if '-' in test_genomes[genome_order[0]] else len(test_genomes[genome_order[0]]) * pattern_length
    serial_time = len(genome_order) * genome_length
    theoretical_min = genome_length

    lines.append("=" * 80)
    lines.append("EFFICIENCY ANALYSIS")
    lines.append("=" * 80)
    lines.append(f"  Serial processing (one at a time) : {serial_time} moves")
    lines.append(f"  Theoretical minimum (perfect)     : {theoretical_min} moves")
    lines.append(f"  Actual result                     : {total_moves} moves")
    if total_moves > 0:
        lines.append(f"  Speedup vs serial                 : {serial_time / total_moves:.2f}x")
        lines.append(f"  Efficiency vs theoretical         : {theoretical_min / total_moves * 100:.1f}%")
    lines.append("=" * 80)

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # =========================================================================
    # CONFIGURATION — edit these to change what runs
    # =========================================================================

    test_genomes = {
        1: 'Ti-Co-Ge-Co-Ti-Co-Ge-Co-Ti-Co-Ge-Co',
        2: 'Mn-Co-Ge-Co-Mn-Co-Ge-Co-Ti-Co-Ge-Co',
        3: 'Mn-Co-Ge-Co-Ti-Co-Ge-Co-Ti-Co-Ge-Co',
        4: 'Mn-Co-Co-Ge-Mn-Co-Co-Ge-Mn-Co-Co-Ge',
        5: 'Mn-Ge-Co-Co-Ti-Ge-Co-Co-Ti-Ge-Co-Co',
        6: 'Mn-Ge-Co-Co-Mn-Ge-Co-Co-Ti-Ge-Co-Co',
        7: 'Ti-Ge-Co-Co-Ti-Ge-Co-Co-Ti-Ge-Co-Co',
        8: 'Mn-Co-Ge-Co-Mn-Co-Ge-Co-Mn-Co-Ge-Co',
        9: 'Mn-Co-Ge-Co-Mn-Co-Ge-Co-Mn-Fe-Ge-Fe',
        10: 'Mn-Co-Ge-Co-Mn-Fe-Ge-Fe-Mn-Fe-Ge-Fe',
        11: 'Mn-Co-Ge-Fe-Mn-Co-Ge-Fe-Mn-Co-Ge-Fe',
        12: 'Mn-Fe-Ge-Co-Mn-Fe-Ge-Co-Mn-Fe-Ge-Co',
        13: 'Mn-Co-Ge-Co-Mn-Co-Ge-Co-Mn-Co-Ge-Fe',
        14: 'Mn-Co-Ge-Co-Mn-Co-Ge-Fe-Mn-Co-Ge-Fe',
        15: 'Mn-Al-Co-Co-Mn-Al-Co-Fe-Mn-Al-Fe-Fe',
        16: 'Mn-Al-Co-Co-Mn-Al-Co-Co-Mn-Al-Fe-Fe',
        17: 'Mn-Al-Fe-Fe-Mn-Al-Fe-Fe-Mn-Al-Fe-Fe'
    }

    '''test_genomes = {
        1: 'Ti-Co-Ge-Co-Ti-Co-Ge',
        2: 'Mn-Co-Ge-Co-Mn-Co-Ge-Co-Ti-Co-Ge-Co',
        3: 'Mn-Co-Ge-Co',
        4: 'Mn-Co-Co-Ge-Mn-Co-Ge',
        5: 'Mn-Ge-Co-Co-Ti-Ge-Co-Co-Ti-Ge-Co-Co',
        6: 'Mn-Ge-Co-Co-Mn-Ge',
        7: 'Ti-Ge-Co-Co-Ti-Ge-Co-Co-Ti-Ge-Co-Co',
        8: 'Mn-Co-Ge-Co-Mn-Co-Ge-Co-Mn-Co-Ge-Co',
        9: 'Mn-Co-Ge-Co-Mn-Co-Ge-Co-Mn-Fe-Ge-Fe',
        10: 'Mn-Co-Ge-Co-Mn-Fe-Ge-Fe-Mn-Fe-Ge-Fe',
        11: 'Mn-Co-Ge-Fe-Mn-Co-Ge-Fe-Mn-Co-Ge-Fe',
        12: 'Mn-Fe-Ge-Co-Mn-Fe-Ge-Co-Mn-Fe-Ge-Co',
    }'''
    # Choose which genomes to use (subset of the dictionary above)
    genome_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # Pattern repetition
    PATTERN_LENGTH = 1

    # Output filename
    OUTPUT_FILE = "epitaxy_results.txt"

    # Set to True to re-run the final simulation at beam_width=3 after ordering.
    # Set to False to skip and just print/save the beam_width=2 result from ordering.
    RUN_FINAL_BEAM3 = True

    # =========================================================================

    scheduler = EpitaxyOrderingAndSimulator(
        genomes=test_genomes,
        genome_order=genome_order,
        pattern_length=PATTERN_LENGTH,
        max_workers=None  # None = auto-detect CPU count
    )

    total_moves, operation_log, stats = scheduler.run(run_final_beam3=RUN_FINAL_BEAM3)

    optimal_order = stats['optimal_order']

    # Print to console
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Optimal order      : {optimal_order}")
    print(f"  Final beam width   : {stats.get('final_beam_width', '?')}")
    print(f"  Total moves        : {total_moves}")
    print(f"  Ordering time      : {stats.get('ordering_time', 0):.3f}s")
    print(f"  Final sim time     : {stats.get('execution_time', 0):.3f}s")
    print(f"  Nodes explored     : {stats.get('nodes_explored', 0):,}")
    print(f"  Beam pruned        : {stats.get('moves_pruned', 0):,}")
    print(f"  Bounds pruned      : {stats.get('bounds_pruned', 0):,}")
    print(f"  Cache hits         : {stats.get('cache_hits', 0):,}")
    print(f"  Max depth          : {stats.get('max_depth_reached', 0)}")

    print_detailed_moves(operation_log, total_moves)

    # Efficiency analysis
    genome_length = len([e for e in test_genomes[genome_order[0]].split('-') if e]) * PATTERN_LENGTH
    serial_time = len(genome_order) * genome_length
    theoretical_min = genome_length
    print(f"\nEFFICIENCY ANALYSIS:")
    print(f"{'=' * 80}")
    print(f"  Serial processing      : {serial_time} moves")
    print(f"  Theoretical minimum    : {theoretical_min} moves")
    print(f"  Actual result          : {total_moves} moves")
    if total_moves > 0:
        print(f"  Speedup vs serial      : {serial_time / total_moves:.2f}x")
        print(f"  Efficiency vs theory   : {theoretical_min / total_moves * 100:.1f}%")
    print(f"{'=' * 80}")

    # Save to file
    output_text = build_output_text(
        genome_order, test_genomes, optimal_order,
        total_moves, operation_log, stats, PATTERN_LENGTH
    )

    with open(OUTPUT_FILE, "w") as f:
        f.write(output_text)

    print(f"\n✓ Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
