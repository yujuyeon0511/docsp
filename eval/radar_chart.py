"""
Radar chart for DocSP-InternVL3_5-8B benchmark results.
Compares DocSP against InternVL2.5-8B baseline.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import glob


def collect_vlmeval_results(work_dir):
    """Collect benchmark results from VLMEvalKit output directory."""
    results = {}
    model_name = "DocSP-InternVL3_5-8B"

    # Map dataset names to display names
    dataset_display = {
        'MMVet': 'MM-Vet',
        'GQA_TestDev_Balanced': 'GQA',
        'VizWiz': 'VizWiz',
        'ScienceQA_TEST': 'SQA-IMG',
        'TextVQA_VAL': 'TextVQA',
        'POPE': 'POPE',
        'MME': 'MME',
        'MMBench_DEV_EN': 'MMBench',
        'SEEDBench_IMG': 'SEED-IMG',
        'LLaVABench': 'LLaVA-Bench',
    }

    # Try to read results from various output formats
    for dataset_key, display_name in dataset_display.items():
        # Look for result files
        patterns = [
            f"{work_dir}/{model_name}/{model_name}_{dataset_key}*.json",
            f"{work_dir}/{model_name}_{dataset_key}*.json",
            f"{work_dir}/*{dataset_key}*acc*.csv",
            f"{work_dir}/*{dataset_key}*result*.json",
        ]

        for pattern in patterns:
            matches = glob.glob(pattern)
            for match in matches:
                try:
                    if match.endswith('.json'):
                        with open(match) as f:
                            data = json.load(f)
                        # Extract score based on dataset type
                        score = extract_score(data, dataset_key)
                        if score is not None:
                            results[display_name] = score
                    elif match.endswith('.csv'):
                        import pandas as pd
                        df = pd.read_csv(match)
                        if 'acc' in df.columns:
                            results[display_name] = df['acc'].iloc[0]
                except Exception as e:
                    print(f"Error reading {match}: {e}")

    return results


def extract_score(data, dataset_key):
    """Extract the main score from benchmark result JSON."""
    if isinstance(data, dict):
        # Common score keys
        for key in ['Overall', 'overall', 'acc', 'Acc', 'accuracy', 'score', 'Score',
                     'total', 'Total', 'average', 'Average', 'avg']:
            if key in data:
                val = data[key]
                if isinstance(val, (int, float)):
                    return float(val)

        # MME uses perception + cognition
        if 'perception' in data and 'cognition' in data:
            return float(data['perception']) + float(data['cognition'])

        # Nested result
        if 'result' in data:
            return extract_score(data['result'], dataset_key)

    return None


def draw_radar_chart(docsp_results, baseline_results=None, output_path='benchmark_radar.png'):
    """Draw radar chart comparing DocSP vs baseline."""
    # Order of benchmarks on the radar
    benchmark_order = [
        'MM-Vet', 'GQA', 'VizWiz', 'SQA-IMG', 'TextVQA',
        'POPE', 'MME', 'MMBench', 'SEED-IMG', 'LLaVA-Bench'
    ]

    # Filter to available benchmarks
    available = [b for b in benchmark_order if b in docsp_results]
    if not available:
        print("No benchmark results available!")
        return

    N = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Normalize scores for display
    # MME is on a different scale (~2000), normalize to percentage
    def normalize_score(benchmark, score):
        if benchmark == 'MME':
            return score / 28.0  # MME full score ~2800, show as percentage-like
        if benchmark == 'LLaVA-Bench':
            return score  # already 0-100
        return score  # most are 0-100 percentages

    # DocSP results
    docsp_values = [normalize_score(b, docsp_results[b]) for b in available]
    docsp_values += docsp_values[:1]
    ax.plot(angles, docsp_values, 'o-', linewidth=2, label='DocSP-InternVL3.5-8B',
            color='#2196F3', markersize=8)
    ax.fill(angles, docsp_values, alpha=0.15, color='#2196F3')

    # Baseline results (if provided)
    if baseline_results:
        baseline_values = [normalize_score(b, baseline_results.get(b, 0)) for b in available]
        baseline_values += baseline_values[:1]
        ax.plot(angles, baseline_values, 's--', linewidth=2, label='InternVL2.5-8B',
                color='#FF5722', markersize=7)
        ax.fill(angles, baseline_values, alpha=0.1, color='#FF5722')

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, size=12, fontweight='bold')

    # Add score annotations
    for i, (angle, val, benchmark) in enumerate(zip(angles[:-1], docsp_values[:-1], available)):
        score_text = f"{docsp_results[benchmark]:.1f}"
        if benchmark == 'MME':
            score_text = f"{docsp_results[benchmark]:.0f}"
        ax.annotate(score_text, xy=(angle, val), fontsize=9,
                    ha='center', va='bottom', color='#2196F3', fontweight='bold')

    if baseline_results:
        for i, (angle, val, benchmark) in enumerate(zip(angles[:-1], baseline_values[:-1], available)):
            if benchmark in baseline_results:
                score_text = f"{baseline_results[benchmark]:.1f}"
                if benchmark == 'MME':
                    score_text = f"{baseline_results[benchmark]:.0f}"
                ax.annotate(score_text, xy=(angle, val), fontsize=9,
                            ha='center', va='top', color='#FF5722')

    # Grid
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8, color='gray')
    ax.yaxis.grid(True, color='gray', alpha=0.3)
    ax.xaxis.grid(True, color='gray', alpha=0.3)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('DocSP-InternVL3.5-8B Benchmark Results', size=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Radar chart saved to: {output_path}")
    plt.close()


# InternVL2.5-8B baseline scores (from official paper/leaderboard)
INTERNVL2_5_8B_BASELINE = {
    'MM-Vet': 62.8,
    'GQA': 64.4,
    'SQA-IMG': 97.4,
    'TextVQA': 77.6,
    'POPE': 88.4,
    'MME': 2344.0,
    'MMBench': 81.7,
    'SEED-IMG': 76.2,
    'LLaVA-Bench': 72.0,
}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', default='/NetDisk/juyeon/VLMEvalKit/outputs/docsp_benchmarks',
                        help='VLMEvalKit output directory')
    parser.add_argument('--output', default='/NetDisk/juyeon/vlm_viz/benchmark_radar.png',
                        help='Output image path')
    parser.add_argument('--manual', type=str, default=None,
                        help='Manual results JSON file (if not using VLMEvalKit output)')
    args = parser.parse_args()

    if args.manual:
        with open(args.manual) as f:
            docsp_results = json.load(f)
    else:
        docsp_results = collect_vlmeval_results(args.work_dir)

    print("DocSP results:", docsp_results)

    if docsp_results:
        draw_radar_chart(docsp_results, INTERNVL2_5_8B_BASELINE, args.output)
    else:
        print("No results found. Check --work-dir path.")
