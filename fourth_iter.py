import itertools
import time
from typing import Dict, List, Tuple, Set, Optional
from copy import deepcopy
import random


class EpitaxyScheduler:
    """
    Solves the crystal genome epitaxy scheduling problem using beam search.
    Combines exhaustive search with intelligent pruning for near-optimal solutions.

    Algorithm Overview:
    1. Generate all possible assignments of genomes to strips
    2. For each assignment, perform beam search DFS
    3. At each level, keep only top-k moves based on heuristic scoring
    4. Use dominated move elimination to safely prune bad choices
    5. Track and return the best solution found

    OPTIMIZATIONS:
    - Pruning bounds: Eliminate branches that can't beat current best
    - State caching: Avoid re-exploring identical states
    - Enhanced move ordering: Better heuristics with look-ahead
    """

    def __init__(self, genomes: Dict[int, str], pattern_length: int = 5, num_strips: int = 5, beam_width: int = 3):
        """
        Initialize the scheduler with genome patterns.

        Args:
            genomes: Dictionary mapping genome ID to pattern string (e.g., "Mn-Co-Al-Co")
            pattern_length: Number of times to repeat each pattern (default 5 for 5-layer patterns)
            num_strips: Number of strips available for deposition (default 5)
            beam_width: Number of top moves to keep at each search level (default 3)
        """
        self.base_genomes = genomes
        self.pattern_length = pattern_length
        self.num_strips = num_strips
        self.beam_width = beam_width

        # Parse and expand patterns to full sequences
        self.genomes = {}
        for genome_id, pattern in genomes.items():
            elements = self._parse_sequence(pattern)
            full_sequence = elements * pattern_length
            self.genomes[genome_id] = full_sequence

        # DFS statistics
        self.nodes_explored = 0
        self.max_depth_reached = 0
        self.moves_pruned = 0
        self.bounds_pruned = 0
        self.cache_hits = 0

        # State cache for memoization
        self.state_cache = {}

        print(f"Initialized with {len(self.genomes)} genomes, each {len(list(self.genomes.values())[0])} layers long")
        print(f"Using {self.num_strips} strips for deposition")
        print(f"Beam width: {self.beam_width} (keeping top-{self.beam_width} moves at each level)")
        for gid, seq in self.genomes.items():
            print(f"Genome_{gid}: {'-'.join(seq)}")

    def _parse_sequence(self, sequence_string: str) -> List[str]:
        """Parse a sequence string into individual elements."""
        if '-' in sequence_string:
            return [element.strip() for element in sequence_string.split('-') if element.strip()]
        else:
            return list(sequence_string)

    def _sequence_to_string(self, sequence: List[str]) -> str:
        """Convert a sequence list back to string representation."""
        if len(sequence) > 0 and len(sequence[0]) > 1:
            return '-'.join(sequence)
        else:
            return ''.join(sequence)

    def _state_to_key(self, strip_states: List[List[str]]) -> str:
        """Convert state to hashable key for caching."""
        return '|'.join(['-'.join(seq) if seq else '' for seq in strip_states])

    def _calculate_lower_bound(self, strip_states: List[List[str]],
                               strip_genomes: List[List[str]],
                               current_moves: int) -> int:
        """Calculate lower bound (optimistic estimate) of remaining moves."""
        max_remaining = 0
        for i in range(self.num_strips):
            if strip_genomes[i] is not None:
                remaining = len(strip_genomes[i]) - len(strip_states[i])
                max_remaining = max(max_remaining, remaining)

        return current_moves + max_remaining

    def get_all_assignments(self) -> List[List[int]]:
        """Generate all possible assignments of genomes to strips."""
        genome_ids = list(self.genomes.keys())

        if len(genome_ids) > self.num_strips:
            return list(itertools.permutations(genome_ids, self.num_strips))
        else:
            assignments = []
            for perm in itertools.permutations(genome_ids):
                assignment = list(perm) + [None] * (self.num_strips - len(perm))
                assignments.append(assignment)
            return assignments

    def get_strips_helped(self, mask_config: List[int], strip_states: List[List[str]],
                          strip_genomes: List[List[str]], element: str) -> List[int]:
        """Get list of strips that actually need this element at their current layer."""
        strips_helped = []
        for strip_idx in mask_config:
            if (strip_genomes[strip_idx] is not None and
                    len(strip_states[strip_idx]) < len(strip_genomes[strip_idx])):
                current_layer = len(strip_states[strip_idx])
                needed_element = strip_genomes[strip_idx][current_layer]
                if needed_element == element:
                    strips_helped.append(strip_idx)
        return strips_helped

    def score_move(self, element: str, mask_config: List[int],
                   strip_states: List[List[str]], strip_genomes: List[List[str]]) -> float:
        """Score a move for beam search pruning with look-ahead."""
        strips_helped = self.get_strips_helped(mask_config, strip_states, strip_genomes, element)

        if len(strips_helped) == 0:
            return -1000

        progress = len(strips_helped)
        efficiency = len(strips_helped) / len(mask_config)

        urgency_bonus = 0
        for strip_idx in strips_helped:
            layers_remaining = len(strip_genomes[strip_idx]) - len(strip_states[strip_idx])
            urgency_bonus += layers_remaining * 0.1

        compactness = 1.0 / len(mask_config)

        lookahead_bonus = 0
        for strip_idx in strips_helped:
            current_layer = len(strip_states[strip_idx])
            if current_layer + 1 < len(strip_genomes[strip_idx]):
                next_element = strip_genomes[strip_idx][current_layer + 1]
                synergy = 0
                for other_strip in range(self.num_strips):
                    if (other_strip != strip_idx and
                            strip_genomes[other_strip] is not None and
                            len(strip_states[other_strip]) < len(strip_genomes[other_strip])):
                        other_layer = len(strip_states[other_strip])
                        if other_layer < len(strip_genomes[other_strip]):
                            if strip_genomes[other_strip][other_layer] == next_element:
                                synergy += 1
                lookahead_bonus += synergy * 0.3

        bottleneck_penalty = 0
        for strip_idx in mask_config:
            if strip_idx not in strips_helped:
                bottleneck_penalty -= 0.5

        score = (progress * 3.0 +
                 efficiency * 2.0 +
                 urgency_bonus +
                 compactness * 0.5 +
                 lookahead_bonus +
                 bottleneck_penalty)

        return score

    def remove_dominated_moves(self, valid_moves: List[Tuple[str, List[int]]],
                               strip_states: List[List[str]],
                               strip_genomes: List[List[str]]) -> List[Tuple[str, List[int]]]:
        """Remove moves that are strictly dominated by other moves."""
        if len(valid_moves) <= 1:
            return valid_moves

        move_info = []
        for element, mask_config in valid_moves:
            strips_helped = set(self.get_strips_helped(mask_config, strip_states, strip_genomes, element))
            move_info.append((element, mask_config, strips_helped))

        non_dominated = []
        for i, (elem_i, mask_i, helped_i) in enumerate(move_info):
            is_dominated = False
            for j, (elem_j, mask_j, helped_j) in enumerate(move_info):
                if i != j:
                    if helped_i < helped_j and helped_i.issubset(helped_j):
                        is_dominated = True
                        break
            if not is_dominated:
                non_dominated.append((elem_i, mask_i))

        pruned_count = len(valid_moves) - len(non_dominated)
        if pruned_count > 0:
            self.moves_pruned += pruned_count

        return non_dominated

    def get_pruned_moves(self, strip_states: List[List[str]],
                         strip_genomes: List[List[str]]) -> List[Tuple[str, List[int]]]:
        """Get moves with multiple pruning strategies."""
        all_moves = self.get_all_valid_moves(strip_states, strip_genomes)

        if len(all_moves) == 0:
            return []

        all_moves = self.remove_dominated_moves(all_moves, strip_states, strip_genomes)

        scored_moves = []
        for element, mask_config in all_moves:
            score = self.score_move(element, mask_config, strip_states, strip_genomes)
            scored_moves.append((score, element, mask_config))

        scored_moves.sort(reverse=True, key=lambda x: x[0])

        beam_limit = min(self.beam_width, len(scored_moves))
        top_moves = [(elem, mask) for _, elem, mask in scored_moves[:beam_limit]]

        pruned_by_beam = len(scored_moves) - beam_limit
        if pruned_by_beam > 0:
            self.moves_pruned += pruned_by_beam

        return top_moves

    def get_all_valid_moves(self, strip_states: List[List[str]], strip_genomes: List[List[str]]) -> List[
        Tuple[str, List[int]]]:
        """Generate ALL valid moves from the current state."""
        active_strips = set()
        for i in range(self.num_strips):
            if (strip_genomes[i] is not None and
                    len(strip_states[i]) < len(strip_genomes[i])):
                active_strips.add(i)

        if not active_strips:
            return []

        element_groups = {}
        for strip_idx in active_strips:
            current_layer = len(strip_states[strip_idx])
            needed_element = strip_genomes[strip_idx][current_layer]

            if needed_element not in element_groups:
                element_groups[needed_element] = []
            element_groups[needed_element].append(strip_idx)

        all_valid_moves = []

        for element, strips_needing in element_groups.items():
            possible_masks = self.get_contiguous_subsets(strip_states, strip_genomes, element)

            for mask_config in possible_masks:
                useful_strips = [s for s in mask_config if s in strips_needing]
                if len(useful_strips) > 0:
                    all_valid_moves.append((element, mask_config))

        return all_valid_moves

    def get_contiguous_subsets(self, strip_states: List[List[str]], strip_genomes: List[List[str]], element: str) -> \
            List[List[int]]:
        """Generate all possible contiguous subsets."""
        valid_strips = []
        for i in range(self.num_strips):
            if (strip_genomes[i] is not None and
                    len(strip_states[i]) < len(strip_genomes[i])):
                current_layer = len(strip_states[i])
                needed_element = strip_genomes[i][current_layer]
                if needed_element == element:
                    valid_strips.append(i)

        if not valid_strips:
            return []

        contiguous_subsets = []

        for start in range(self.num_strips):
            for end in range(start, self.num_strips):
                subset = list(range(start, end + 1))

                valid_subset = True
                contains_needed_strip = False

                for strip_idx in subset:
                    if strip_idx in valid_strips:
                        contains_needed_strip = True
                    else:
                        if (strip_genomes[strip_idx] is not None and
                                len(strip_states[strip_idx]) >= len(strip_genomes[strip_idx])):
                            valid_subset = False
                            break
                        elif (strip_genomes[strip_idx] is not None and
                              len(strip_states[strip_idx]) < len(strip_genomes[strip_idx])):
                            valid_subset = False
                            break

                if valid_subset and contains_needed_strip:
                    contiguous_subsets.append(subset)

        return contiguous_subsets

    def _is_complete(self, strip_states: List[List[str]], strip_genomes: List[List[str]]) -> bool:
        """Check if all strips have completed their sequences."""
        for i in range(self.num_strips):
            if strip_genomes[i] is not None:
                if len(strip_states[i]) < len(strip_genomes[i]):
                    return False
        return True

    def _dfs_search(self, strip_states: List[List[str]], strip_genomes: List[List[str]],
                    current_moves: int, current_log: List[str], depth: int,
                    best_known: Optional[int] = None) -> Tuple[Optional[int], Optional[List[str]]]:
        """Beam search depth-first search with pruning bounds and state caching."""
        self.nodes_explored += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)

        lower_bound = self._calculate_lower_bound(strip_states, strip_genomes, current_moves)

        if best_known is not None and lower_bound >= best_known:
            self.bounds_pruned += 1
            return (None, None)

        state_key = self._state_to_key(strip_states)
        if state_key in self.state_cache:
            cached_remaining = self.state_cache[state_key]
            cached_total = current_moves + cached_remaining
            self.cache_hits += 1
            if best_known is not None and cached_total >= best_known:
                self.bounds_pruned += 1
                return (None, None)

        if self._is_complete(strip_states, strip_genomes):
            self.state_cache[state_key] = 0
            return (current_moves, current_log.copy())

        valid_moves = self.get_pruned_moves(strip_states, strip_genomes)

        if not valid_moves:
            return (None, None)

        best_total_moves = best_known
        best_log = None

        for element, mask_config in valid_moves:
            new_strip_states = deepcopy(strip_states)
            new_log = current_log.copy()

            useful_strips = []
            for strip_idx in mask_config:
                if strip_genomes[strip_idx] is not None and len(new_strip_states[strip_idx]) < len(
                        strip_genomes[strip_idx]):
                    new_strip_states[strip_idx].append(element)
                    if len(strip_states[strip_idx]) < len(strip_genomes[strip_idx]):
                        if strip_genomes[strip_idx][len(strip_states[strip_idx])] == element:
                            useful_strips.append(strip_idx)

            move_num = current_moves + 1
            new_log.append(f"Move {move_num}: Deposit {element} on strips {mask_config} (needed by: {useful_strips})")

            state_strs = []
            for i in range(self.num_strips):
                state_str = self._sequence_to_string(new_strip_states[i])
                state_strs.append(f"Strip {i}: {state_str}")
            new_log.append(f"  State: {' | '.join(state_strs)}")

            result_moves, result_log = self._dfs_search(
                new_strip_states,
                strip_genomes,
                current_moves + 1,
                new_log,
                depth + 1,
                best_total_moves
            )

            if result_moves is not None:
                if best_total_moves is None or result_moves < best_total_moves:
                    best_total_moves = result_moves
                    best_log = result_log

        if best_total_moves is not None:
            remaining_moves = best_total_moves - current_moves
            self.state_cache[state_key] = remaining_moves

        return (best_total_moves, best_log)

    def simulate_deposition(self, assignment: List[int]) -> Tuple[int, List[str]]:
        """Simulate the deposition process using beam search DFS."""
        strip_states = [[] for _ in range(self.num_strips)]
        strip_genomes = [None] * self.num_strips

        for i, genome_id in enumerate(assignment):
            if genome_id is not None:
                strip_genomes[i] = self.genomes[genome_id]

        self.nodes_explored = 0
        self.max_depth_reached = 0
        self.moves_pruned = 0
        self.bounds_pruned = 0
        self.cache_hits = 0
        self.state_cache = {}

        min_moves, operation_log = self._dfs_search(strip_states, strip_genomes, 0, [], 0, None)

        if min_moves is None:
            return (float('inf'), ["Error: No valid deposition sequence found"])

        return (min_moves, operation_log)

    def find_optimal_assignment(self) -> Tuple[List[int], int, List[str], float, Dict]:
        """Find the optimal assignment of genomes to strips using beam search."""
        start_time = time.time()

        all_assignments = self.get_all_assignments()
        best_assignment = None
        min_moves = float('inf')
        best_log = []
        total_nodes_explored = 0
        total_max_depth = 0
        total_moves_pruned = 0
        total_bounds_pruned = 0
        total_cache_hits = 0

        print(f"\nEvaluating {len(all_assignments)} possible assignments...")

        for i, assignment in enumerate(all_assignments):
            print(
                f"  Assignment {i + 1}/{len(all_assignments)}: {['Genome_' + str(g) if g is not None else 'Empty' for g in assignment]}",
                end=" ")

            moves, log = self.simulate_deposition(assignment)

            print(
                f"-> {moves} moves ({self.nodes_explored} nodes, {self.moves_pruned} beam, {self.bounds_pruned} bound, {self.cache_hits} cache)")

            total_nodes_explored += self.nodes_explored
            total_max_depth = max(total_max_depth, self.max_depth_reached)
            total_moves_pruned += self.moves_pruned
            total_bounds_pruned += self.bounds_pruned
            total_cache_hits += self.cache_hits

            if moves < min_moves:
                min_moves = moves
                best_assignment = assignment
                best_log = log

        end_time = time.time()
        execution_time = end_time - start_time

        stats = {
            'total_nodes_explored': total_nodes_explored,
            'total_moves_pruned': total_moves_pruned,
            'total_bounds_pruned': total_bounds_pruned,
            'total_cache_hits': total_cache_hits,
            'max_depth_reached': total_max_depth,
            'assignments_evaluated': len(all_assignments),
            'avg_nodes_per_assignment': total_nodes_explored / len(all_assignments),
            'avg_moves_pruned_per_assignment': total_moves_pruned / len(all_assignments),
            'avg_bounds_pruned_per_assignment': total_bounds_pruned / len(all_assignments),
            'avg_cache_hits_per_assignment': total_cache_hits / len(all_assignments)
        }

        return best_assignment, min_moves, best_log, execution_time, stats


def load_genomes_from_file(filename: str, selected_lines: List[int]) -> Dict[int, str]:
    """Load specific genomes from file based on line numbers."""
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        genomes = {}
        for i, line_num in enumerate(selected_lines, 1):
            if 1 <= line_num <= len(lines):
                genomes[i] = lines[line_num - 1]
            else:
                print(f"Warning: Line {line_num} is out of range (file has {len(lines)} lines)")

        return genomes

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return {}
    except Exception as e:
        print(f"Error reading file: {e}")
        return {}


def print_detailed_moves(operation_log: List[str], min_moves: int):
    """Print detailed move-by-move instructions."""
    print(f"\n{'=' * 80}")
    print(f"DETAILED DEPOSITION INSTRUCTIONS ({min_moves} total moves)")
    print(f"{'=' * 80}\n")

    move_count = 0
    for line in operation_log:
        if line.startswith("Move"):
            move_count += 1
            # Extract move details
            parts = line.split(":")
            move_info = parts[1].strip()
            print(f"\n{'─' * 80}")
            print(f"STEP {move_count}:")
            print(f"  ACTION: {move_info}")
        elif line.startswith("  State:"):
            # Print state after the move
            state_info = line.replace("  State: ", "")
            strips = state_info.split(" | ")
            print(f"  RESULT:")
            for strip in strips:
                print(f"    {strip}")
    print(f"\n{'─' * 80}")
    print(f"✓ All genomes completed in {min_moves} deposition moves!")
    print(f"{'=' * 80}\n")


def main():
    """Main function to demonstrate the scheduler."""
    print("=" * 80)
    print("CRYSTAL GENOME EPITAXY SCHEDULER")
    print("=" * 80)

    # Test genomes as specified
    test_genomes = {
        1: 'Mn-Co-Al-Co-Mn-Co-Al-Co-Mn-Co-Al-Co',
        2: 'Mn-Co-Co-Al-Mn-Co-Co-Al-Mn-Co-Co-Al',
        3: 'Mn-Al-Co-Co-Mn-Al-Co-Co-Mn-Al-Co-Co',
        4: 'Mn-Mn-Co-Al-Mn-Mn-Co-Al-Mn-Mn-Co-Al',
        5: 'Mn-Al-Co-Mn-Mn-Al-Co-Mn-Mn-Al-Co-Mn'
    }

    print("\nTest Genomes:")
    for gid, pattern in test_genomes.items():
        print(f"  Genome {gid}: {pattern}")

    scheduler = EpitaxyScheduler(
        genomes=test_genomes,
        pattern_length=1,  # Already fully specified
        num_strips=5,
        beam_width=3
    )

    best_assignment, min_moves, operation_log, exec_time, stats = scheduler.find_optimal_assignment()

    print(f"\n{'=' * 80}")
    print("OPTIMIZATION RESULTS:")
    print(f"{'=' * 80}")
    print(f"Best Assignment: {['Genome_' + str(g) if g is not None else 'Empty' for g in best_assignment]}")
    print(f"Minimum Moves Required: {min_moves}")
    print(f"Execution Time: {exec_time:.3f} seconds")

    print(f"\nGenome-to-Strip Mapping:")
    for strip_idx, genome_id in enumerate(best_assignment):
        if genome_id is not None:
            print(f"  Strip {strip_idx} ← Genome {genome_id}: {test_genomes[genome_id]}")
        else:
            print(f"  Strip {strip_idx} ← Empty")

    print(f"\nSearch Statistics:")
    print(f"  Total nodes explored: {stats['total_nodes_explored']:,}")
    print(f"  Moves pruned (beam search): {stats['total_moves_pruned']:,}")
    print(f"  Moves pruned (bounds): {stats['total_bounds_pruned']:,}")
    print(f"  Cache hits: {stats['total_cache_hits']:,}")
    print(f"  Max depth reached: {stats['max_depth_reached']}")
    print(f"  Assignments evaluated: {stats['assignments_evaluated']}")
    print(f"  Avg nodes per assignment: {stats['avg_nodes_per_assignment']:.1f}")

    # Print detailed move instructions
    print_detailed_moves(operation_log, min_moves)

    # Print efficiency analysis
    genome_length = len(scheduler.genomes[1])
    serial_time = len(test_genomes) * genome_length
    theoretical_min = genome_length

    print(f"\nEFFICIENCY ANALYSIS:")
    print(f"{'=' * 80}")
    print(f"  Serial processing (one at a time): {serial_time} moves")
    print(f"  Theoretical minimum (perfect parallel): {theoretical_min} moves")
    print(f"  Actual optimal solution: {min_moves} moves")
    print(f"  Speedup vs serial: {serial_time / min_moves:.2f}x")
    print(f"  Efficiency vs theoretical: {theoretical_min / min_moves * 100:.1f}%")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
