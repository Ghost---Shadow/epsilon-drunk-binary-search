import random
import statistics
import numpy as np
import scipy.stats as stats_scipy
import csv
import json
from typing import List, Tuple
from tqdm import tqdm
from collections import Counter


def drunk_binary_search(
    arr: List[int], target: int, epsilon_range: float = 0.1
) -> Tuple[int, int]:
    """
    Epsilon drunk binary search where the midpoint calculation has a random offset.

    Args:
        arr: Sorted list to search
        target: Value to find
        epsilon_range: Maximum deviation from 0.5 (e.g., 0.1 means epsilon in [-0.1, 0.1])

    Returns:
        Tuple of (index if found or -1, number of comparisons)
    """
    low = 0
    high = len(arr) - 1
    comparisons = 0

    while low <= high:
        # The "drunk" part: instead of 0.5, use 0.5 + epsilon
        epsilon = random.uniform(-epsilon_range, epsilon_range)
        drunk_ratio = 0.5 + epsilon

        # Clamp to reasonable bounds to avoid going outside array
        drunk_ratio = max(0.1, min(0.9, drunk_ratio))

        # Calculate drunk midpoint
        mid = int(low + (high - low) * drunk_ratio)
        comparisons += 1

        if arr[mid] == target:
            return mid, comparisons
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1, comparisons


def regular_binary_search(arr: List[int], target: int) -> Tuple[int, int]:
    """
    Regular binary search for comparison.

    Returns:
        Tuple of (index if found or -1, number of comparisons)
    """
    low = 0
    high = len(arr) - 1
    comparisons = 0

    while low <= high:
        mid = (low + high) // 2
        comparisons += 1

        if arr[mid] == target:
            return mid, comparisons
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1, comparisons


def generate_array_with_gini(size: int, target_gini: float = 0.0) -> List[int]:
    """
    Generate an array targeting a specific Gini coefficient for gap distribution.
    Uses deterministic approach for speed.
    
    Args:
        size: Number of elements
        target_gini: Target Gini coefficient (0.0 = perfectly even gaps, higher = more uneven)
    
    Returns:
        Sorted list with gap distribution matching target Gini coefficient
    """
    if target_gini == 0.0:
        # Perfectly even gaps - evenly spaced array
        return list(range(size))
    
    # Fast deterministic approach: create gaps with desired inequality
    gaps = []
    num_gaps = size - 1
    
    if target_gini <= 0.15:
        # Low inequality: mostly 1s with some 2s
        num_large = int(target_gini * num_gaps * 3)  # Scale factor for desired inequality
        gaps = [1] * (num_gaps - num_large) + [2] * num_large
    elif target_gini <= 0.3:
        # Medium inequality: mix of 1s and larger gaps
        num_large = int(target_gini * num_gaps * 2)
        gaps = [1] * (num_gaps - num_large) + [3] * num_large
    else:
        # High inequality: mix of 1s and much larger gaps
        num_large = int(target_gini * num_gaps)
        gap_size = min(8, int(2 + target_gini * 10))  # Scale gap size with target Gini
        gaps = [1] * (num_gaps - num_large) + [gap_size] * num_large
    
    # Shuffle to avoid patterns and build array
    random.shuffle(gaps)
    
    array = [0]
    for gap in gaps:
        array.append(array[-1] + gap)
    
    return array


def calculate_gini_coefficient(gaps: List[int]) -> float:
    """Calculate Gini coefficient for a list of gaps."""
    if not gaps or len(gaps) < 2:
        return 0.0
    
    sorted_gaps = sorted(gaps)
    n = len(sorted_gaps)
    cumsum = sum(sorted_gaps)
    
    if cumsum == 0:
        return 0.0
    
    gini = (2 * sum((i + 1) * gap for i, gap in enumerate(sorted_gaps))) / (n * cumsum) - (n + 1) / n
    return max(0.0, gini)  # Ensure non-negative


def analyze_array_drunkness(arr: List[int]) -> dict:
    """
    Calculate the Gini coefficient of an array's gap distribution.
    This is the only metric we use for array drunkness.
    
    Returns:
        Dictionary with Gini coefficient as the sole drunkness metric
    """
    if len(arr) < 2:
        return {"gini_coefficient": 0.0}

    # Calculate gaps between consecutive elements
    gaps = [arr[i + 1] - arr[i] for i in range(len(arr) - 1)]
    
    # Calculate Gini coefficient using our helper function
    gini = calculate_gini_coefficient(gaps)
    
    return {
        "gini_coefficient": gini
    }


def benchmark_search_algorithms(
    arr_size: int = 10000,
    num_trials: int = 1000,
    epsilon_values: List[float] = None,
    target_gini: float = 0.0,
) -> dict:
    """
    Benchmark different epsilon values against regular binary search.
    Now supports "drunk" arrays with uneven spacing.
    """
    if epsilon_values is None:
        epsilon_values = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    print(
        f"Benchmarking with array size {arr_size}, {num_trials} trials per epsilon..."
    )
    print(
        f"Testing epsilon range: {min(epsilon_values):.2f} to {max(epsilon_values):.2f}"
    )
    print("  (negative = undershoot, positive = overshoot)")

    if target_gini > 0:
        print(f"🥴 Target array Gini coefficient: {target_gini:.2f}")
    else:
        print("😐 Array: perfectly sober (evenly spaced)")

    results = {eps: [] for eps in epsilon_values}
    regular_results = []

    for trial in tqdm(range(num_trials), desc="Running trials", unit="trial"):
        # Generate array with target Gini coefficient
        arr = generate_array_with_gini(arr_size, target_gini)

        # Pick a random target that exists in the array
        target_idx = random.randint(0, arr_size - 1)
        target = arr[target_idx]

        # Test regular binary search
        _, regular_comps = regular_binary_search(arr, target)
        regular_results.append(regular_comps)

        # Test each epsilon value
        for epsilon in epsilon_values:
            if epsilon == 0.0:
                # For epsilon = 0, just use regular binary search
                results[epsilon].append(regular_comps)
            else:
                _, drunk_comps = drunk_binary_search(arr, target, abs(epsilon))
                results[epsilon].append(drunk_comps)

    # Calculate statistics with confidence intervals
    stats = {}
    regular_avg = statistics.mean(regular_results)
    regular_std = statistics.stdev(regular_results)

    # 95% confidence interval using t-distribution approximation
    confidence_level = 0.95
    alpha = 1 - confidence_level

    for epsilon in epsilon_values:
        data = results[epsilon]
        n = len(data)
        avg_comps = statistics.mean(data)
        std_comps = statistics.stdev(data)
        improvement = ((regular_avg - avg_comps) / regular_avg) * 100
        drunkness_level = abs(epsilon)

        # Calculate 95% confidence interval
        t_critical = stats_scipy.t.ppf(1 - alpha / 2, df=n - 1)
        margin_of_error = t_critical * (std_comps / np.sqrt(n))
        ci_lower = avg_comps - margin_of_error
        ci_upper = avg_comps + margin_of_error

        # Statistical significance test vs regular binary search
        # Two-sample t-test (assuming unequal variances)
        if epsilon != 0.0:
            t_stat, p_value = stats_scipy.ttest_ind(
                data, regular_results, equal_var=False
            )
            is_significant = p_value < 0.05
        else:
            t_stat, p_value, is_significant = 0.0, 1.0, False

        stats[epsilon] = {
            "avg_comparisons": avg_comps,
            "std_comparisons": std_comps,
            "improvement_pct": improvement,
            "drunkness_level": drunkness_level,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "margin_of_error": margin_of_error,
            "sample_size": n,
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant": is_significant,
            "all_results": data,
        }

    stats["regular_avg"] = regular_avg
    stats["regular_std"] = regular_std
    stats["total_trials"] = num_trials
    stats["target_gini"] = target_gini
    return stats


def find_balmer_peak(
    arr_size: int = 10000,
    num_trials: int = 1000,
    target_gini: float = 0.0,
) -> Tuple[float, dict]:
    """
    Find the optimal epsilon value (the "Balmer peak").
    Tests both positive (overshoot) and negative (undershoot) epsilon values.
    Now supports drunk arrays with uneven spacing.
    """
    print("🍺 Searching for the Balmer Peak...")

    # Test expanded range of epsilon values for comprehensive search
    epsilon_values = np.arange(-0.3, 0.31, 0.02)  # From -0.3 to +0.3, step 0.02
    stats = benchmark_search_algorithms(
        arr_size, num_trials, epsilon_values, target_gini
    )

    # Find the epsilon with best average performance
    valid_epsilons = [eps for eps in epsilon_values]
    best_epsilon = min(valid_epsilons, key=lambda eps: stats[eps]["avg_comparisons"])

    return best_epsilon, stats


def save_results_to_csv(all_scenarios_results: list, filename: str = "epsilon_drunk_search_results.csv"):
    """Save all scenario results to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'scenario_name', 'array_gini_coefficient', 'arr_size', 'total_trials',
            'regular_avg', 'regular_std', 'epsilon', 'avg_comparisons', 'std_comparisons',
            'improvement_pct', 'ci_lower', 'ci_upper', 'margin_of_error', 'sample_size',
            't_statistic', 'p_value', 'is_significant', 'drunkness_level'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for scenario_result in all_scenarios_results:
            scenario_name = scenario_result['scenario_name']
            stats = scenario_result['stats']
            
            # Write metadata for each epsilon
            epsilons = [eps for eps in stats.keys() if isinstance(eps, (int, float))]
            
            for eps in sorted(epsilons):
                eps_stats = stats[eps]
                drunkness_analysis = scenario_result.get('drunkness_analysis', {})
                row = {
                    'scenario_name': scenario_name,
                    'array_gini_coefficient': drunkness_analysis.get('gini_coefficient', 0.0),
                    'arr_size': scenario_result.get('arr_size', 0),
                    'total_trials': stats.get('total_trials', 0),
                    'regular_avg': stats.get('regular_avg', 0.0),
                    'regular_std': stats.get('regular_std', 0.0),
                    'epsilon': eps,
                    'avg_comparisons': eps_stats['avg_comparisons'],
                    'std_comparisons': eps_stats['std_comparisons'],
                    'improvement_pct': eps_stats['improvement_pct'],
                    'ci_lower': eps_stats['ci_lower'],
                    'ci_upper': eps_stats['ci_upper'],
                    'margin_of_error': eps_stats['margin_of_error'],
                    'sample_size': eps_stats['sample_size'],
                    't_statistic': eps_stats['t_statistic'],
                    'p_value': eps_stats['p_value'],
                    'is_significant': eps_stats['is_significant'],
                    'drunkness_level': eps_stats['drunkness_level']
                }
                writer.writerow(row)
    
    print(f"📊 Results saved to '{filename}'")








# Main analysis
if __name__ == "__main__":
    print("🔍 Epsilon Drunk Binary Search Analysis")
    print("=" * 60)

    # Grid search across different Gini coefficient levels - comprehensive search
    gini_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    
    # Create scenarios for different target Gini coefficients
    scenarios = []
    for target_gini in gini_levels:
        if target_gini == 0.0:
            scenario_name = "Sober Array"
        else:
            scenario_name = f"Array (Gini={target_gini:.1f})"
            
        scenarios.append({
            "name": scenario_name,
            "target_gini": target_gini
        })

    all_scenarios_results = []

    for i, scenario in enumerate(
        tqdm(scenarios, desc="Testing scenarios", unit="scenario")
    ):
        print(f"\n{'='*60}")
        print(f"🧪 SCENARIO {i+1}: {scenario['name']}")
        print(f"{'='*60}")

        # Find the Balmer peak for this scenario
        # Balance between resolution and speed
        arr_size, num_trials = 5000, 100
        
        best_epsilon, stats = find_balmer_peak(
            arr_size=arr_size,
            num_trials=num_trials,
            target_gini=scenario["target_gini"],
        )
        regular_avg = stats["regular_avg"]

        # Generate a sample array for drunkness analysis
        sample_arr = generate_array_with_gini(arr_size, scenario["target_gini"])
        
        # Analyze the drunkness of the generated array
        drunkness_analysis = analyze_array_drunkness(sample_arr)

        # Store results for CSV export
        scenario_result = {
            'scenario_name': scenario['name'],
            'arr_size': arr_size,
            'stats': stats,
            'drunkness_analysis': drunkness_analysis
        }
        all_scenarios_results.append(scenario_result)

        print(
            f"\n🎯 Results for {scenario['name']} (n={stats['total_trials']} trials per epsilon):"
        )
        print(
            f"Regular binary search: {regular_avg:.2f} ± {stats['regular_std']:.2f} comparisons"
        )
        print(f"Optimal epsilon (Balmer peak): {best_epsilon:.4f}")
        print(
            f"  -> {'Overshoot' if best_epsilon > 0 else 'Undershoot' if best_epsilon < 0 else 'Perfect aim'}"
        )
        print(f"  -> Search drunkness level: |ε| = {abs(best_epsilon):.4f}")

        best_stats = stats[best_epsilon]
        print(
            f"Best drunk search: {best_stats['avg_comparisons']:.2f} ± {best_stats['std_comparisons']:.2f} comparisons"
        )
        print(
            f"  -> 95% CI: [{best_stats['ci_lower']:.2f}, {best_stats['ci_upper']:.2f}]"
        )
        print(f"  -> Improvement: {best_stats['improvement_pct']:.2f}%")
        print(
            f"  -> Statistical significance: {'YES' if best_stats['is_significant'] else 'NO'} (p={best_stats['p_value']:.4f})"
        )
        
        # Display array drunkness analysis
        print(f"\n🧪 Array Drunkness Analysis:")
        print(f"  -> Actual Gini Coefficient: {drunkness_analysis['gini_coefficient']:.3f}")
        print(f"  -> Target Gini Coefficient: {scenario['target_gini']:.3f}")

        # Show some interesting epsilon values with confidence intervals
        print(f"\n📊 Performance at different search epsilon levels (with 95% CI):")
        interesting_epsilons = [
            -0.2,
            -0.1,
            -0.05,
            0.0,
            0.05,
            0.1,
            0.15,
            0.2,
            best_epsilon,
        ]
        for eps in sorted(set(interesting_epsilons)):
            if eps in stats and eps not in [
                "regular_avg",
                "regular_std",
                "total_trials",
                "target_gini",
            ]:
                s = stats[eps]
                direction = (
                    "undershoot" if eps < 0 else "overshoot" if eps > 0 else "sober"
                )
                significance = "*" if s["is_significant"] else " "
                print(
                    f"  ε = {eps:+.3f} ({direction:>9}): {s['avg_comparisons']:.2f} ± {s['std_comparisons']:.2f} "
                    f"[{s['ci_lower']:.2f}, {s['ci_upper']:.2f}] ({s['improvement_pct']:+.2f}%){significance}"
                )

        # Test with a smaller example to show actual searches
        print(f"\n🧪 Example searches with optimal epsilon ({best_epsilon:.4f}):")
        test_arr = generate_array_with_gini(100, scenario["target_gini"])

        for j in range(3):
            target_idx = random.randint(0, 99)
            target = test_arr[target_idx]
            regular_idx, regular_comps = regular_binary_search(test_arr, target)
            drunk_idx, drunk_comps = drunk_binary_search(
                test_arr, target, abs(best_epsilon)
            )
            print(
                f"  Target {target} (idx {target_idx}): Regular={regular_comps} vs Drunk={drunk_comps} comparisons"
            )

        print(
            f"\n* = statistically significant difference from regular binary search (p < 0.05)"
        )
        print(f"CI = Confidence Interval, ± = Standard Deviation")

    # Save all results to CSV
    save_results_to_csv(all_scenarios_results)

    print(f"\n{'='*60}")
    print("🏁 ANALYSIS COMPLETE!")
    print("Results saved to 'epsilon_drunk_search_results.csv'")
    print("Use the plotting script to generate visualizations.")
    print(f"{'='*60}")
