#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paired significance tests for sanitized evaluation result JSON files."""
import argparse
import json
import math
from collections import defaultdict
import numpy as np


def load_results(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["detailed_results"]


def make_occurrence_keys(results):
    counts = defaultdict(int)
    keyed = {}
    for r in results:
        base_key = (r.get("source"), r.get("expected"))
        counts[base_key] += 1
        key = (base_key[0], base_key[1], counts[base_key])
        keyed[key] = r
    return keyed


def pair_results(baseline, best):
    if len(baseline) == len(best):
        ordered_ok = all(
            b.get("source") == s.get("source") and b.get("expected") == s.get("expected")
            for b, s in zip(baseline, best)
        )
        if ordered_ok:
            return list(zip(baseline, best))

    b_map = make_occurrence_keys(baseline)
    s_map = make_occurrence_keys(best)
    common_keys = sorted(set(b_map) & set(s_map))

    dropped = len(baseline) - len(common_keys)
    if dropped > 0:
        print(f"WARNING: {dropped} unmatched pairs dropped. Check whether both JSONs use the same test set.")

    return [(b_map[k], s_map[k]) for k in common_keys]


def is_answered(r):
    return r.get("mae_sec") is not None


def exact_at(r, tol=30):
    mae = r.get("mae_sec")
    return mae is not None and mae <= tol


def large_error_rate(results, threshold=300):
    """Percentage of answered predictions more than threshold seconds off (default 5 min)."""
    answered = [r for r in results if is_answered(r)]
    if not answered:
        return None
    return sum(r["mae_sec"] > threshold for r in answered) / len(answered)


def mean_mae_common_answered(pairs):
    vals = []
    for b, s in pairs:
        if is_answered(b) and is_answered(s):
            vals.append((b["mae_sec"], s["mae_sec"]))
    if not vals:
        return None, None, None
    b_vals = np.array([x[0] for x in vals], dtype=float)
    s_vals = np.array([x[1] for x in vals], dtype=float)
    return float(np.mean(b_vals)), float(np.mean(s_vals)), float(np.mean(s_vals - b_vals))


def exact_mcnemar(pairs, tol=30):
    """
    McNemar exact binomial test for Exact@tol.
    b_count = baseline wrong, best correct
    c_count = baseline correct, best wrong
    """
    b_count = 0
    c_count = 0
    for base, best in pairs:
        base_ok = exact_at(base, tol)
        best_ok = exact_at(best, tol)
        if (not base_ok) and best_ok:
            b_count += 1
        elif base_ok and (not best_ok):
            c_count += 1

    n = b_count + c_count
    if n == 0:
        return b_count, c_count, 1.0

    k = min(b_count, c_count)
    cdf = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    p_value = min(1.0, 2.0 * cdf)
    return b_count, c_count, p_value


def wilcoxon_or_sign_test(pairs):
    """
    Paired Wilcoxon signed-rank test for MAE on commonly answered cases.
    diff = best_mae - baseline_mae (negative = best is better).
    Also returns effect size r = Z / sqrt(N).
    """
    diffs = []
    for b, s in pairs:
        if is_answered(b) and is_answered(s):
            diffs.append(float(s["mae_sec"]) - float(b["mae_sec"]))

    diffs = np.array(diffs, dtype=float)
    diffs_nonzero = diffs[diffs != 0]

    if len(diffs_nonzero) == 0:
        return len(diffs), None, None, None, "no_nonzero_differences"

    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(diffs_nonzero, alternative="two-sided")
        n = len(diffs_nonzero)
        # Effect size r = Z / sqrt(N); approximate Z from Wilcoxon statistic
        # scipy returns W statistic; use normal approximation for effect size
        mean_w = n * (n + 1) / 4
        std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (stat - mean_w) / std_w if std_w > 0 else 0.0
        r_effect = abs(z) / math.sqrt(n)
        return len(diffs), float(stat), float(p), float(r_effect), "wilcoxon"
    except Exception:
        pos = int(np.sum(diffs_nonzero > 0))
        neg = int(np.sum(diffs_nonzero < 0))
        n = pos + neg
        k = min(pos, neg)
        cdf = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
        p = min(1.0, 2.0 * cdf)
        return len(diffs), None, float(p), None, "sign_test_fallback"


def cluster_bootstrap(pairs, n_boot=10000, seed=42, tols=(30, 60, 120)):
    """
    Cluster bootstrap by transcript/source.
    Resamples source files, not individual queries.
    Computes CIs for Exact@tol (all tolerances), MAE, Answered, and large-error rate.
    """
    rng = np.random.default_rng(seed)

    groups = defaultdict(list)
    for base, best in pairs:
        groups[base.get("source")].append((base, best))

    sources = list(groups.keys())

    exact_diffs = {tol: [] for tol in tols}
    mae_diffs = []
    answered_diffs = []
    large_error_diffs = []

    for _ in range(n_boot):
        sampled_sources = rng.choice(sources, size=len(sources), replace=True)
        sample = []
        for src in sampled_sources:
            sample.extend(groups[src])

        for tol in tols:
            base_exact = np.mean([exact_at(b, tol) for b, _ in sample])
            best_exact = np.mean([exact_at(s, tol) for _, s in sample])
            exact_diffs[tol].append(best_exact - base_exact)

        base_answered = np.mean([is_answered(b) for b, _ in sample])
        best_answered = np.mean([is_answered(s) for _, s in sample])
        answered_diffs.append(best_answered - base_answered)

        common = [
            (b["mae_sec"], s["mae_sec"])
            for b, s in sample
            if is_answered(b) and is_answered(s)
        ]
        if common:
            b_mae = np.mean([x[0] for x in common])
            s_mae = np.mean([x[1] for x in common])
            mae_diffs.append(s_mae - b_mae)

        base_ler = np.mean([b["mae_sec"] > 300 for b, _ in sample if is_answered(b)]) if any(is_answered(b) for b, _ in sample) else 0
        best_ler = np.mean([s["mae_sec"] > 300 for _, s in sample if is_answered(s)]) if any(is_answered(s) for _, s in sample) else 0
        large_error_diffs.append(best_ler - base_ler)

    def ci(vals):
        vals = np.array(vals, dtype=float)
        return (
            float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)),
        )

    result = {
        "n_sources": len(sources),
        "mae_diff_mean_ci": ci(mae_diffs),
        "answered_diff_mean_ci": ci(answered_diffs),
        "large_error_diff_mean_ci": ci(large_error_diffs),
    }
    for tol in tols:
        result[f"exact{tol}_diff_mean_ci"] = ci(exact_diffs[tol])

    return result


def summarize(name, results, tols=(30, 60, 120)):
    n = len(results)
    answered = sum(is_answered(r) for r in results)
    maes = [r["mae_sec"] for r in results if is_answered(r)]
    ler = large_error_rate(results, threshold=300)

    print(f"\n{name}")
    print("=" * 60)
    print(f"N: {n}")
    print(f"Answered: {answered}/{n} ({answered/n*100:.1f}%)")
    for tol in tols:
        exact = sum(exact_at(r, tol) for r in results)
        print(f"Exact@{tol}s: {exact}/{n} ({exact/n*100:.1f}%)")
    if maes:
        print(f"Mean MAE (answered): {np.mean(maes):.1f}s")
        print(f"Median AE (answered): {np.median(maes):.1f}s")
        print(f"P90 AE (answered): {np.percentile(maes, 90):.1f}s")
        print(f"P95 AE (answered): {np.percentile(maes, 95):.1f}s")
    if ler is not None:
        print(f"Large-error rate (>5 min): {ler*100:.1f}%")


def build_json_output(pairs, boot, tols=(30, 60, 120)):
    """Build a structured dict for easy paper-table extraction."""
    out = {"cluster_bootstrap": {"n_sources": boot["n_sources"]}}

    for tol in tols:
        mean, lo, hi = boot[f"exact{tol}_diff_mean_ci"]
        b_count, c_count, p = exact_mcnemar(pairs, tol)
        out[f"exact{tol}s"] = {
            "delta_pp": round(mean * 100, 2),
            "ci_95_pp": [round(lo * 100, 2), round(hi * 100, 2)],
            "mcnemar_p": round(p, 6),
            "baseline_wrong_best_correct": b_count,
            "baseline_correct_best_wrong": c_count,
        }

    mae_mean, mae_lo, mae_hi = boot["mae_diff_mean_ci"]
    n_common, stat, p_mae, r_effect, test_name = wilcoxon_or_sign_test(pairs)
    out["mae"] = {
        "delta_s": round(mae_mean, 1),
        "ci_95_s": [round(mae_lo, 1), round(mae_hi, 1)],
        "test": test_name,
        "p": round(p_mae, 6) if p_mae is not None else None,
        "effect_size_r": round(r_effect, 3) if r_effect is not None else None,
        "n_common_answered": n_common,
    }

    ans_mean, ans_lo, ans_hi = boot["answered_diff_mean_ci"]
    out["answered"] = {
        "delta_pp": round(ans_mean * 100, 2),
        "ci_95_pp": [round(ans_lo * 100, 2), round(ans_hi * 100, 2)],
    }

    ler_mean, ler_lo, ler_hi = boot["large_error_diff_mean_ci"]
    out["large_error_rate_5min"] = {
        "delta_pp": round(ler_mean * 100, 2),
        "ci_95_pp": [round(ler_lo * 100, 2), round(ler_hi * 100, 2)],
    }

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Significance tests and cluster bootstrap for topic-to-timestamp evaluation."
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline JSON results.")
    parser.add_argument("--best", required=True, help="Path to best system JSON results.")
    parser.add_argument("--tol", type=int, default=30, help="Primary Exact@tol threshold (default: 30).")
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap iterations (default: 10000).")
    parser.add_argument("--json-out", default=None, help="Optional path to write structured JSON output.")
    args = parser.parse_args()

    TOLS = (30, 60, 120)

    baseline = load_results(args.baseline)
    best = load_results(args.best)
    pairs = pair_results(baseline, best)

    print(f"Paired cases: {len(pairs)}")
    if len(pairs) != len(baseline) or len(pairs) != len(best):
        print("WARNING: Pair count differs from one or both files.")

    summarize("Baseline", [b for b, _ in pairs], tols=TOLS)
    summarize("Best", [s for _, s in pairs], tols=TOLS)

    # Paired MAE summary
    b_mae, s_mae, mae_diff = mean_mae_common_answered(pairs)
    print("\nPaired MAE (commonly answered cases)")
    print("=" * 60)
    if b_mae is not None:
        print(f"Baseline mean MAE: {b_mae:.1f}s")
        print(f"Best mean MAE:     {s_mae:.1f}s")
        print(f"Difference (best - baseline): {mae_diff:.1f}s  [negative = best is better]")

    # McNemar for all tolerances
    print("\nMcNemar exact test")
    print("=" * 60)
    for tol in TOLS:
        b_count, c_count, p = exact_mcnemar(pairs, tol)
        sig = "significant" if p < 0.05 else "not significant"
        print(f"Exact@{tol}s  |  baseline wrong / best correct: {b_count}  |  baseline correct / best wrong: {c_count}  |  p={p:.6f}  [{sig}]")

    # Wilcoxon for MAE
    n_common, stat, p_mae, r_effect, test_name = wilcoxon_or_sign_test(pairs)
    print("\nPaired MAE test")
    print("=" * 60)
    print(f"Test: {test_name}  |  Common answered cases: {n_common}")
    if stat is not None:
        print(f"Statistic: {stat:.3f}")
    if p_mae is not None:
        sig = "significant" if p_mae < 0.05 else "not significant"
        print(f"p-value: {p_mae:.6f}  [{sig}]")
    if r_effect is not None:
        print(f"Effect size r: {r_effect:.3f}  [small<0.1 | medium<0.3 | large>=0.5]")

    # Cluster bootstrap
    print(f"\nCluster bootstrap (n_boot={args.n_boot}, clustered by transcript/source)")
    print("=" * 60)
    boot = cluster_bootstrap(pairs, n_boot=args.n_boot, tols=TOLS)
    print(f"Number of transcript clusters: {boot['n_sources']}")

    for tol in TOLS:
        mean, lo, hi = boot[f"exact{tol}_diff_mean_ci"]
        sig = "CI excludes 0" if not (lo <= 0 <= hi) else "CI includes 0"
        print(f"Δ Exact@{tol}s:  {mean*100:+.2f} pp  95% CI [{lo*100:.2f}, {hi*100:.2f}]  [{sig}]")

    mae_mean, mae_lo, mae_hi = boot["mae_diff_mean_ci"]
    sig = "CI excludes 0" if not (mae_lo <= 0 <= mae_hi) else "CI includes 0"
    print(f"Δ MAE:          {mae_mean:+.1f}s   95% CI [{mae_lo:.1f}, {mae_hi:.1f}]  [{sig}]  [negative = best is better]")

    ans_mean, ans_lo, ans_hi = boot["answered_diff_mean_ci"]
    sig = "CI excludes 0" if not (ans_lo <= 0 <= ans_hi) else "CI includes 0"
    print(f"Δ Answered:     {ans_mean*100:+.2f} pp  95% CI [{ans_lo*100:.2f}, {ans_hi*100:.2f}]  [{sig}]")

    ler_mean, ler_lo, ler_hi = boot["large_error_diff_mean_ci"]
    sig = "CI excludes 0" if not (ler_lo <= 0 <= ler_hi) else "CI includes 0"
    print(f"Δ Large-error rate (>5min): {ler_mean*100:+.2f} pp  95% CI [{ler_lo*100:.2f}, {ler_hi*100:.2f}]  [{sig}]  [negative = best is better]")

    # Optional JSON output
    if args.json_out:
        structured = build_json_output(pairs, boot, tols=TOLS)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2)
        print(f"\nStructured results written to: {args.json_out}")


if __name__ == "__main__":
    main()