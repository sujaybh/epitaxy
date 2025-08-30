# Crystal Genome Epitaxy Scheduler

A computational optimization tool for parallel molecular beam epitaxy (MBE) deposition of crystal genomes - atomic-scale patterns deposited layer by layer to create novel materials with designed properties.

## Project Overview

### The Physical Process

This project addresses the scheduling optimization for a specialized epitaxial deposition system:

- **Crystal Genomes**: 12-layer atomic sequences that encode material properties, similar to how DNA encodes biological information
- **Epitaxy**: Controlled atomic layer-by-layer deposition process in ultra-high vacuum
- **Parallel Processing**: 5 substrate strips processed simultaneously under movable masks
- **Elements**: 4 different atomic species (A, B, C, D) available for deposition

### The Engineering Challenge

The system consists of:
- **Vacuum Chamber**: Ultra-high vacuum environment for clean deposition
- **Substrate Holder**: Horizontal slide with 5 adjacent processing strips [A, B, C, D, E]
- **Movable Masks**: Two masks that can expose any contiguous subset of strips
- **Element Sources**: Metal drums with electron beam evaporators for each element
- **Time Constraint**: Each atomic layer deposition takes time T

### The Computational Problem

Given ~1000 target crystal genome sequences, optimize:
1. **Assignment**: Which genomes to process on which strips
2. **Scheduling**: Order of deposition operations to minimize total time
3. **Mask Management**: Optimal positioning to maximize useful parallel work

**Key Constraint**: Masks can only expose contiguous strips (e.g., [B,C,D] or [A,B] but not [A,C,E])

## Scientific Background

### Crystal Genomes
Crystal genomes represent a revolutionary approach to materials design where atomic-scale layering patterns determine macroscopic properties. Each 12-layer sequence creates specific:
- Electronic band structures
- Optical properties  
- Mechanical characteristics
- Catalytic behaviors

### Epitaxial Growth
Molecular beam epitaxy enables precise control of:
- **Layer thickness**: Single atomic layer precision
- **Composition**: Element-by-element control
- **Interface quality**: Atomically sharp transitions
- **Pattern fidelity**: Exact reproduction of designed sequences

## Installation and Setup

### Prerequisites
- Python 3.7+
- Standard library only (no external dependencies)

### Files Required
- `epitaxy_scheduler.py` - Main algorithm implementation
- `genomes.txt` - Database of 100 sample crystal genome patterns
- `README.md` - This documentation

### Quick Start
```bash
# Clone or download the project files
# Ensure genomes.txt is in the same directory as the Python script

python epitaxy_scheduler.py
```

## Usage

### Basic Operation
The script runs with default genome selections from lines [1, 15, 30, 45, 60] of genomes.txt:

```python
python epitaxy_scheduler.py
```

### Custom Genome Selection
Modify the `selected_genome_lines` list in `main()` to choose different genomes:

```python
# Select genomes from specific lines (1-indexed)
selected_genome_lines = [1, 25, 50, 75, 100]  # Lines from genomes.txt
selected_genome_lines = [10, 20, 30, 40, 50]  # Different combination
selected_genome_lines = [5, 15, 25, 35, 45]   # Another set
```

### Understanding the Output

The script provides comprehensive analysis:

1. **Genome Loading**: Shows which patterns were loaded
2. **Assignment Evaluation**: Tests all possible strip arrangements
3. **Optimal Result**: Best assignment and minimum operations needed
4. **Detailed Log**: Step-by-step deposition operations
5. **Performance Metrics**: Speedup analysis vs. serial processing

### Example Output
```
=== Crystal Genome Epitaxy Scheduler ===
Selected genome lines: [1, 15, 30, 45, 60]
Loaded genomes: {1: 'ABCDABCDABCD', 2: 'ABABABABABAB', ...}

Initialized with 5 genomes, each 12 layers long
Genome_1: ABCDABCDABCD
Genome_2: ABABABABABAB
...

Evaluating 120 possible assignments...
Assignment 1: ['Genome_1', 'Genome_2', 'Genome_3', 'Genome_4', 'Genome_5'] -> 45 moves
Assignment 2: ['Genome_1', 'Genome_2', 'Genome_3', 'Genome_5', 'Genome_4'] -> 43 moves
...

=== RESULTS ===
Best assignment: ['Genome_2', 'Genome_1', 'Genome_4', 'Genome_3', 'Genome_5']
Strip positions: ['Strip_0: Genome_2', 'Strip_1: Genome_1', 'Strip_2: Genome_4', 'Strip_3: Genome_3', 'Strip_4: Genome_5']
Minimum moves required: 38

=== SPEEDUP ANALYSIS ===
serial_processing_moves: 60.00
actual_optimal_moves: 38.00
speedup_vs_serial: 1.58
```

## Algorithm Details

### Core Algorithm: Greedy Constraint Satisfaction with Exhaustive Assignment Search

**Phase 1: Assignment Generation**
- Generates all permutations of genome-to-strip assignments
- Handles cases with fewer than 5 genomes (partial assignments)

**Phase 2: Deposition Simulation**
- Simulates layer-by-layer deposition for each assignment
- Tracks progress on each strip independently
- Groups strips needing the same element at each step

**Phase 3: Mask Optimization**
- For each element needed, finds all valid contiguous mask positions
- Optimizes for maximum useful work: `useful_strips / total_exposed_strips`
- Selects operations that advance the most strips simultaneously

**Phase 4: Performance Evaluation**
- Compares all assignments to find global optimum
- Provides detailed operation logs for the best solution
- Calculates speedup metrics vs. serial processing

### Complexity Analysis
- **Time Complexity**: O(n! × L × E) where n=genomes, L=layers, E=elements
- **Space Complexity**: O(n × L) for storing sequences and states
- **Scalability**: Suitable for small-scale problems (≤10 genomes)

## Genome Database (genomes.txt)

The included database contains 100 carefully designed 12-layer patterns featuring:

- **Periodic Patterns**: Regular repeating sequences (ABCDABCDABCD)
- **Grouped Elements**: Clustered same-element regions (AAAABBBBCCCC)
- **Alternating Patterns**: High-frequency element switching (ABABABABABAB)
- **Complex Sequences**: Mixed patterns with varying periodicities
- **Shifted Variants**: Rotated versions of base patterns

### Pattern Categories
- **Lines 1-20**: Simple periodic and grouped patterns
- **Lines 21-40**: Alternating and mixed frequency patterns  
- **Lines 41-60**: Complex sequences with forward/reverse elements
- **Lines 61-80**: Shifted and rotated pattern variants
- **Lines 81-100**: Advanced mixed patterns for challenging optimization

## Research Applications

### Materials Science
- **Novel 2D Materials**: Design quantum wells and superlattices
- **Electronic Devices**: Create custom band gap materials
- **Optical Components**: Engineer photonic crystals and metamaterials
- **Catalytic Surfaces**: Design specific active site arrangements

### Process Optimization
- **Throughput Maximization**: Reduce manufacturing time for material libraries
- **Resource Efficiency**: Minimize element usage and chamber time
- **Quality Control**: Ensure precise pattern reproduction across all strips
- **Scalability**: Enable high-throughput materials discovery

## Extending the Project

### Increasing Complexity
```python
# Modify for longer patterns
scheduler = EpitaxyScheduler(genomes, pattern_length=2)  # 24-layer sequences

# Add more elements
# Update genomes.txt with patterns using A, B, C, D, E, F

# Increase strip count
# Modify num_strips parameter for larger substrate holders
```

### Advanced Algorithms
The current implementation provides a foundation for more sophisticated approaches:
- **Machine Learning**: Train models to predict optimal assignments
- **Genetic Algorithms**: Evolve solutions for larger problem instances
- **Integer Programming**: Formulate as optimization problem for guaranteed optimality
- **Heuristic Search**: Implement A* or beam search for better scalability

### Real-World Integration
- **Hardware Interface**: Connect to actual MBE chamber control systems
- **Process Monitoring**: Integrate with deposition rate and quality sensors
- **Database Integration**: Connect to materials property databases
- **Experimental Validation**: Compare predictions with actual deposition results

## Performance Expectations

### Typical Speedups
- **Well-matched genomes**: 3-4x speedup vs. serial processing
- **Diverse genomes**: 1.5-2x speedup vs. serial processing  
- **Identical genomes**: Up to 5x speedup (theoretical maximum)

### Factors Affecting Performance
- **Pattern Similarity**: More similar patterns enable better synchronization
- **Element Distribution**: Balanced element usage improves parallelization
- **Mask Repositioning Time**: Hardware delays can reduce effective speedup
- **Strip Utilization**: Partial assignments reduce maximum possible speedup

## Contributing

This project addresses a real materials science challenge. Contributions welcome for:
- **Algorithm Improvements**: Better optimization strategies
- **Scalability**: Handling larger genome databases
- **Visualization**: Tools for understanding deposition patterns
- **Hardware Integration**: Interfaces to real epitaxy systems
- **Validation**: Experimental verification of predictions

## Technical Notes

### Assumptions
- Instantaneous mask repositioning (can be modified to include repositioning time)
- Perfect deposition uniformity across exposed strips
- No cross-contamination between elements
- Deterministic deposition rates

### Limitations
- Exponential complexity limits practical size to ~10 genomes
- Current implementation assumes contiguous masking only
- No consideration of thermal effects or chamber conditioning

## License and Citation

This computational tool was developed for materials science research applications. When using this software in research, please cite the optimization approach and acknowledge the epitaxial deposition context.

---

*This project represents the intersection of computational optimization and advanced materials synthesis, enabling the systematic exploration of designed atomic-scale architectures.*
