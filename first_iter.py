import itertools
from typing import Dict, List, Tuple, Set
from copy import deepcopy


class EpitaxyScheduler:
    """
    Solves the crystal genome epitaxy scheduling problem using constraint satisfaction.

    Algorithm Overview:
    1. Generate all possible assignments of genomes to strips
    2. For each assignment, simulate the deposition process
    3. Use a greedy approach to minimize total operations at each step
    4. Track mask positions and find optimal contiguous subsets
    5. Return the assignment with minimum total moves
    """

    def __init__(self, genomes: Dict[int, str], pattern_length: int = 5):
        """
        Initialize the scheduler with genome patterns.

        Args:
            genomes: Dictionary mapping genome ID to pattern string
            pattern_length: Number of times to repeat each pattern (default 5 for 5-layer patterns)
        """
        self.base_genomes = genomes
        self.pattern_length = pattern_length
        self.num_strips = 5

        # Expand patterns to full sequences
        self.genomes = {}
        for genome_id, pattern in genomes.items():
            self.genomes[genome_id] = pattern * pattern_length

        print(f"Initialized with {len(self.genomes)} genomes, each {len(list(self.genomes.values())[0])} layers long")
        for gid, seq in self.genomes.items():
            print(f"Genome_{gid}: {seq}")

    def get_all_assignments(self) -> List[List[int]]:
        """Generate all possible assignments of genomes to strips."""
        genome_ids = list(self.genomes.keys())

        if len(genome_ids) > self.num_strips:
            # If more genomes than strips, generate combinations
            return list(itertools.permutations(genome_ids, self.num_strips))
        else:
            # If fewer or equal genomes, use all and pad with None
            assignments = []
            for perm in itertools.permutations(genome_ids):
                assignment = list(perm) + [None] * (self.num_strips - len(perm))
                assignments.append(assignment)
            return assignments

    def get_contiguous_subsets(self, positions: Set[int]) -> List[List[int]]:
        """
        Generate all possible contiguous subsets that cover the given positions.
        This represents different mask configurations.
        """
        if not positions:
            return [[]]

        sorted_pos = sorted(positions)
        min_pos, max_pos = min(sorted_pos), max(sorted_pos)

        contiguous_subsets = []

        # Generate all contiguous ranges that include all required positions
        for start in range(min_pos + 1):  # Can start from 0 to min_pos
            for end in range(max_pos, self.num_strips):  # Must extend to at least max_pos
                subset = list(range(start, end + 1))
                if all(pos in subset for pos in positions):
                    contiguous_subsets.append(subset)

        return contiguous_subsets

    def simulate_deposition(self, assignment: List[int]) -> Tuple[int, List[str]]:
        """
        Simulate the deposition process for a given assignment.

        Returns:
            Tuple of (total_moves, operation_log)
        """
        # Initialize strip states
        strip_states = [''] * self.num_strips
        strip_genomes = [None] * self.num_strips

        # Assign genomes to strips
        for i, genome_id in enumerate(assignment):
            if genome_id is not None:
                strip_genomes[i] = self.genomes[genome_id]

        moves = 0
        operation_log = []

        # Continue until all strips are complete
        while True:
            # Find which strips need more material
            active_strips = set()
            for i in range(self.num_strips):
                if (strip_genomes[i] is not None and
                        len(strip_states[i]) < len(strip_genomes[i])):
                    active_strips.add(i)

            if not active_strips:
                break  # All strips complete

            # Group strips by the element they need next
            element_groups = {}
            for strip_idx in active_strips:
                current_layer = len(strip_states[strip_idx])
                needed_element = strip_genomes[strip_idx][current_layer]

                if needed_element not in element_groups:
                    element_groups[needed_element] = []
                element_groups[needed_element].append(strip_idx)

            # For each element, find the optimal mask configuration
            best_move = None
            best_benefit = 0

            for element, strips_needing in element_groups.items():
                strips_set = set(strips_needing)
                possible_masks = self.get_contiguous_subsets(strips_set)

                for mask_config in possible_masks:
                    # Count how many useful depositions this would make
                    useful_strips = [s for s in mask_config if s in strips_set]
                    benefit = len(useful_strips)
                    cost = len(mask_config)  # Total strips exposed (some might be wasted)

                    # Prioritize moves that maximize useful work
                    efficiency = benefit / cost if cost > 0 else 0

                    if benefit > best_benefit or (benefit == best_benefit and efficiency > best_move[3]):
                        best_move = (element, mask_config, useful_strips, efficiency)
                        best_benefit = benefit

            if best_move is None:
                break

            # Execute the best move
            element, mask_config, useful_strips, efficiency = best_move

            # Update strip states
            for strip_idx in useful_strips:
                strip_states[strip_idx] += element

            moves += 1
            operation_log.append(f"Move {moves}: Deposit {element} on strips {mask_config} (useful: {useful_strips})")

            # Debug: show current state
            state_str = " | ".join([f"Strip {i}: {strip_states[i]}" for i in range(self.num_strips)])
            operation_log.append(f"  State: {state_str}")

        return moves, operation_log

    def find_optimal_assignment(self) -> Tuple[List[int], int, List[str]]:
        """
        Find the optimal assignment of genomes to strips.

        Returns:
            Tuple of (best_assignment, min_moves, operation_log)
        """
        all_assignments = self.get_all_assignments()
        best_assignment = None
        min_moves = float('inf')
        best_log = []

        print(f"\nEvaluating {len(all_assignments)} possible assignments...")

        for i, assignment in enumerate(all_assignments):
            moves, log = self.simulate_deposition(assignment)

            #print(f"Assignment {i + 1}: {['Genome_' + str(g) if g is not None else 'Empty' for g in assignment]} -> {moves} moves")

            if moves < min_moves:
                min_moves = moves
                best_assignment = assignment
                best_log = log

        return best_assignment, min_moves, best_log

    def calculate_theoretical_speedup(self) -> Dict[str, float]:
        """Calculate theoretical speedup metrics."""
        # Serial processing time (each genome processed individually)
        genome_length = len(list(self.genomes.values())[0])
        serial_time = len(self.genomes) * genome_length

        # Theoretical minimum (if perfect parallelization were possible)
        theoretical_min = genome_length  # If all genomes were identical

        # Find actual optimal
        _, optimal_moves, _ = self.find_optimal_assignment()

        return {
            'serial_processing_moves': serial_time,
            'theoretical_minimum_moves': theoretical_min,
            'actual_optimal_moves': optimal_moves,
            'speedup_vs_serial': serial_time / optimal_moves,
            'efficiency_vs_theoretical': theoretical_min / optimal_moves
        }


def load_genomes_from_file(filename: str, selected_lines: List[int]) -> Dict[int, str]:
    """
    Load specific genomes from file based on line numbers.

    Args:
        filename: Path to the genomes file
        selected_lines: List of line numbers (1-indexed) to load

    Returns:
        Dictionary mapping genome ID to sequence
    """
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        genomes = {}
        for i, line_num in enumerate(selected_lines, 1):
            if 1 <= line_num <= len(lines):
                genomes[i] = lines[line_num - 1]  # Convert to 0-indexed
            else:
                print(f"Warning: Line {line_num} is out of range (file has {len(lines)} lines)")

        return genomes

    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        print("Using default test genomes instead...")
        return {
            1: 'ABCDABCDABCD',
            2: 'AABBCCDDAABB',
            3: 'ABCDDCBABCDD',
            4: 'CADBCADBCADB',
            5: 'BCDABCDABCDA'
        }


def main():
    """Example usage and testing."""

    # Option 1: Load from file (specify which line numbers you want)
    # Example: Load genomes from lines 1, 15, 30, 45, 60
    selected_genome_lines = [4, 6, 7, 9, 85]
    test_genomes = load_genomes_from_file('genomes.txt', selected_genome_lines)

    # Option 2: Use hardcoded genomes (uncomment to use instead)
    # test_genomes = {
    #     1: 'ABCDABCDABCD',
    #     2: 'AABBCCDDAABB',
    #     3: 'ABCDDCBABCDD',
    #     4: 'CADBCADBCADB',
    #     5: 'BCDABCDABCDA'
    # }

    print("=== Crystal Genome Epitaxy Scheduler ===")
    print(f"Selected genome lines: {selected_genome_lines}")
    print(f"Loaded genomes: {test_genomes}")

    # Create scheduler with pattern length of 1 (since patterns are already 12 layers long)
    scheduler = EpitaxyScheduler(test_genomes, pattern_length=1)

    # Find optimal assignment
    print("\n=== Finding Optimal Assignment ===")
    best_assignment, min_moves, operation_log = scheduler.find_optimal_assignment()

    print(f"\n=== RESULTS ===")
    print(f"Best assignment: {['Genome_' + str(g) if g is not None else 'Empty' for g in best_assignment]}")
    strip_assignments = []
    for i in range(5):
        if best_assignment[i] is not None:
            strip_assignments.append(f"Strip_{i}: Genome_{best_assignment[i]}")
        else:
            strip_assignments.append(f"Strip_{i}: Empty")
    print(f"Strip positions: {strip_assignments}")
    print(f"Minimum moves required: {min_moves}")

    print(f"\n=== DETAILED OPERATION LOG ===")
    for operation in operation_log:
        print(operation)

    # Calculate speedup metrics
    print(f"\n=== SPEEDUP ANALYSIS ===")
    metrics = scheduler.calculate_theoretical_speedup()
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

    print(f"\n=== ALGORITHM EXPLANATION ===")
    print("""
    Algorithm: Greedy Constraint Satisfaction with Exhaustive Assignment Search

    1. ASSIGNMENT PHASE:
       - Generate all possible ways to assign genomes to the 5 strips
       - This includes permutations and partial assignments

    2. SIMULATION PHASE (for each assignment):
       - Track current state of each strip (how many layers deposited)
       - At each step, identify which strips need material
       - Group strips by the element they need next
       - Find optimal mask configuration that maximizes useful work

    3. MASK OPTIMIZATION:
       - For each element needed, find all contiguous strip combinations
       - Choose the configuration that maximizes (useful_strips / total_exposed_strips)
       - This balances doing useful work vs. wasting deposition on unnecessary strips

    4. GREEDY SELECTION:
       - At each step, choose the move that deposits material on the most
         strips that actually need that element
       - Break ties by choosing moves with higher efficiency ratios

    5. TERMINATION:
       - Continue until all strips have completed their full genome sequences
       - Return total number of deposition operations needed

    The algorithm finds the globally optimal assignment by exhaustive search over
    assignments, then uses a greedy approach for the scheduling subproblem.
    """)


if __name__ == "__main__":
    main()
