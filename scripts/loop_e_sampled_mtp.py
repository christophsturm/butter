#!/usr/bin/env python3
# Copyright 2026 Eric Kryski (@ekryski)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ============================================================================
# Loop E (Nondeterministic) statistical-acceptance harness — sampled MTP
# spec decode.
#
# Methodology: RST-AI "Loop E Nondeterministic"
#   (~/Documents/obsidian/Self Study/RST/RST-AI-SINGLE-FILE.md, Cheat Sheet +
#   "## Loop E Nondeterministic" section). Loop E's premise: when the system
#   itself is stochastic (temperature > 0 sampling), "repeat the test" is
#   not meaningful — only *characterizing a distribution* is. It requires:
#     step 2 — fix and record the sampling config verbatim,
#     step 4 — establish a baseline distribution (here: spec-off, the
#              existing accepted sampler),
#     step 5 — run N trials under test (here: spec-on, sampled MTP),
#     step 6 — compare distributions via a NAMED significance test, not a
#              raw diff of two numbers,
#     step 8 — report pass rate / variance / significance / N — never a
#              single pass/fail, and never trust an n=1 run as evidence.
#
# THE CLAIM UNDER TEST
#   Sampled MTP spec decode (BUTTER_QWEN_MTP_SPEC=1 + STATE_REWIND +
#   BATCHED_LOCKSTEP + GAMMA=3, temperature 0.6, match-accept scheme) is
#   LOSSLESS: every emitted token is a true draw from the SAME filtered
#   distribution the no-spec sampler draws from. Two prior seeded digest
#   matches are an existence proof (n=2), not a statistical gate — this
#   harness is the gate that existence proof was missing.
#
#   If sampled spec decode is lossless, the spec-on and spec-off arms are
#   two independent samples from IDENTICAL underlying distributions (first
#   token id, and — since maxTokens acts as a shared cap — tokens-to-stop).
#   A significance test failing to reject "same distribution" (p not tiny)
#   is consistent with (does not prove, per Loop E step 6/8 framing) the
#   lossless claim. A significant divergence (p < alpha) means the arms are
#   NOT drawing from the same distribution — i.e. the accept/match scheme is
#   biasing which tokens get emitted. That is exactly the bug class
#   rejection sampling exists to prevent: a spec-decode accept rule that
#   silently distorts the target distribution instead of just accelerating
#   unbiased draws from it.
#
# STATISTICS (formulas below; see also inline docstrings)
#   1. FIRST-TOKEN distribution — contingency table of {arm} x {token
#      category}, category = distinct first-token ids that recur >=2 times
#      pooled across both arms, tail pooled into "other". Test: Pearson
#      chi-square via scipy.stats.chi2_contingency if scipy is importable;
#      else a stdlib G-test (likelihood-ratio chi-square) fallback:
#          G = 2 * sum_i O_i * ln(O_i / E_i)     (Wilks 1938)
#          df = (rows - 1) * (cols - 1)
#          p  = chi2.sf(G, df)   [regularized upper incomplete gamma Q(df/2, G/2)]
#      Lossless => same first-token distribution in both arms => p NOT tiny.
#
#   2. LENGTH distribution — tokens-to-stop (== tokens emitted before EOS or
#      the --max-tokens cap) per run, per arm. Test: Mann-Whitney U via
#      scipy.stats.mannwhitneyu if importable; else a stdlib rank-sum
#      implementation with tie correction and normal approximation:
#          U1 = R1 - n1*(n1+1)/2                  (R1 = sum of ranks, arm 1)
#          mu_U = n1*n2/2
#          sigma_U = sqrt(n1*n2/12 * ((n1+n2+1) - tie_correction))
#          z = (U1 - mu_U) / sigma_U;  p = 2*(1 - Phi(|z|))   (two-sided)
#      Lossless => same length distribution in both arms => p NOT tiny.
#
#   3. PER-ARM DETERMINISM CONTROL — same seed run twice within an arm must
#      produce an identical digest (sha256 of the full generated-token-id
#      sequence). This is the "is seeding even working" precondition: if it
#      fails, the seeded-sample assumption behind (1) and (2) is void and
#      the whole harness gives no information. STOP on failure — do not
#      proceed to the statistical sweep.
#
# VERDICT: PASS at alpha=0.01 requires BOTH p_first_token > alpha AND
# p_length > alpha (fail to reject "same distribution" on both axes).
# FAIL (p < alpha on either axis) means: the sampled-MTP accept scheme is
# demonstrably NOT drawing from the same distribution as the reference
# sampler — the lossless claim is falsified at this alpha, and per RST-AI
# Loop E step 7 the specific failing trials should be pulled and read
# directly (clustered failure = one systematic mode; scattered = the test
# under-detecting, reconsider N) before shipping any fix.
#
# RELEASE GATE: if this harness PASSes, it becomes the required release
# gate for any future change to the sampled-MTP accept/match scheme
# (Qwen3x SpecDecode.generateSampledMTP and anything touching
# Sampling.buildFilteredWeights along that path). Re-run before merging any
# such change; a regression here blocks the merge same as a red CI gate.
#
# OPERATIONAL CONSTRAINTS (repo/session mandate — see harness code):
#   - Foreground only. No background-and-wait.
#   - One heavy (model) job at a time — runs are strictly sequential.
#   - `pkill -x butter` after every subprocess, success or failure.
#   - `swift test` is NEVER invoked here (this harness drives the built
#     `butter` CLI binary directly, not the Swift test suite).
#   - python3 only (this file).
#   - Pre-flight arithmetic before every model-loading run: peak RSS
#     estimate must be < 60% of `vm_stat`'s (free + inactive) pages.
#   - Per-run watchdog: 300s (subprocess timeout; on expiry the run is
#     recorded as a failed trial, the process is killed, and the sweep
#     continues — a hang is data, not grounds to silently retry).
# ============================================================================

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import scipy.stats as _scipy_stats  # type: ignore

    HAVE_SCIPY = True
except ImportError:
    _scipy_stats = None
    HAVE_SCIPY = False

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = REPO_ROOT / ".build" / "release" / "butter"
DEFAULT_MODEL = str(Path.home() / "models" / "qwen36-27b-4bit-mlx")

# Env vars for the two arms (Sources/ButterSwift/Generation/Generate.swift +
# MTPSpecDecode.swift — grep'd verbatim, not guessed):
SPEC_ON_ENV = {
    "BUTTER_QWEN_MTP_SPEC": "1",
    "BUTTER_QWEN_MTP_STATE_REWIND": "1",
    "BUTTER_QWEN_MTP_BATCHED_LOCKSTEP": "1",
    "BUTTER_QWEN_MTP_GAMMA": "3",
}
SPEC_OFF_ENV = {
    # Explicitly unset (not just "0") so a stray inherited env var can't
    # leak spec decode on into the control arm.
    "BUTTER_QWEN_MTP_SPEC": None,
    "BUTTER_QWEN_MTP_STATE_REWIND": None,
    "BUTTER_QWEN_MTP_BATCHED_LOCKSTEP": None,
    "BUTTER_QWEN_MTP_GAMMA": None,
}

TOKENS_RE = re.compile(r"generated:\s*(\d+)\s*tokens")
TOKEN_IDS_RE = re.compile(r"token ids:\s*\[([^\]]*)\]")

PEAK_MODEL_GB = 20.0  # ~20GB peak per the task's model sizing note
MEM_HEADROOM_FRACTION = 0.60  # peak must stay under 60% of free+inactive
WATCHDOG_SECONDS = 300


# ─── Pre-flight memory arithmetic ──────────────────────────────────────────


def vm_stat_free_inactive_bytes() -> tuple[int, int]:
    """Parse `vm_stat` and return (free_bytes, inactive_bytes).

    macOS-only (this harness runs on the M5 Max Mac). Page size is read
    from vm_stat's own header line rather than hardcoded, since it can
    differ (4096 vs 16384) across Apple Silicon generations/configs.
    """
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    m = re.search(r"page size of (\d+) bytes", out)
    if not m:
        raise RuntimeError(f"could not parse page size from vm_stat output:\n{out}")
    page_size = int(m.group(1))

    def pages(label: str) -> int:
        mm = re.search(rf"{re.escape(label)}:\s*(\d+)\.", out)
        if not mm:
            raise RuntimeError(f"could not parse '{label}' from vm_stat output:\n{out}")
        return int(mm.group(1))

    free = pages("Pages free") * page_size
    inactive = pages("Pages inactive") * page_size
    return free, inactive


def preflight_check(peak_gb: float = PEAK_MODEL_GB) -> tuple[bool, str]:
    """Return (ok, message). ok=False means DO NOT launch the model."""
    free_b, inactive_b = vm_stat_free_inactive_bytes()
    headroom_b = free_b + inactive_b
    peak_b = peak_gb * 1e9
    limit_b = MEM_HEADROOM_FRACTION * headroom_b
    ok = peak_b < limit_b
    msg = (
        f"preflight: free+inactive={headroom_b / 1e9:.1f}GB, "
        f"60%={limit_b / 1e9:.1f}GB, peak_est={peak_gb:.1f}GB -> "
        f"{'OK' if ok else 'BLOCK'}"
    )
    return ok, msg


# ─── Running a single trial ────────────────────────────────────────────────


@dataclass
class TrialResult:
    arm: str
    seed: int
    first_token: int | None
    second_token: int | None
    n_tokens: int | None
    digest: str | None
    elapsed_s: float
    ok: bool
    note: str = ""


def build_env(arm_env: dict) -> dict:
    env = os.environ.copy()
    for k, v in arm_env.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


def pkill_butter() -> None:
    # -x = exact process-name match only (never -f — see CLAUDE.md gotcha:
    # `pkill -f` on a cmdline containing the pattern string as a *literal
    # argument* (e.g. inside this very script's argv) can self-kill the
    # invoking shell instead of the target).
    subprocess.run(["pkill", "-x", "butter"], capture_output=True)


def run_trial(
    binary: Path,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    arm: str,
    arm_env: dict,
    watchdog_s: int = WATCHDOG_SECONDS,
) -> TrialResult:
    ok, msg = preflight_check()
    print(f"  [{arm} seed={seed}] {msg}", file=sys.stderr)
    if not ok:
        return TrialResult(
            arm=arm, seed=seed, first_token=None, second_token=None,
            n_tokens=None, digest=None, elapsed_s=0.0, ok=False,
            note="preflight BLOCK",
        )

    cmd = [
        str(binary),
        "generate",
        "--model",
        model,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
        "--print-token-ids",
        "--no-streaming",
    ]
    env = build_env(arm_env)

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=watchdog_s
        )
        elapsed = time.monotonic() - start
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        pkill_butter()
        return TrialResult(
            arm=arm, seed=seed, first_token=None, second_token=None,
            n_tokens=None, digest=None, elapsed_s=elapsed, ok=False,
            note=f"WATCHDOG TIMEOUT after {watchdog_s}s",
        )
    finally:
        # Defensive: always reap any stray butter process after a run,
        # success or failure, per the one-heavy-job-at-a-time mandate.
        pkill_butter()

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-500:]
        return TrialResult(
            arm=arm, seed=seed, first_token=None, second_token=None,
            n_tokens=None, digest=None, elapsed_s=elapsed, ok=False,
            note=f"exit={proc.returncode} tail={tail!r}",
        )

    stdout = proc.stdout
    n_tokens_m = TOKENS_RE.search(stdout)
    ids_m = TOKEN_IDS_RE.search(stdout)
    if not n_tokens_m or not ids_m:
        tail = stdout[-500:]
        return TrialResult(
            arm=arm, seed=seed, first_token=None, second_token=None,
            n_tokens=None, digest=None, elapsed_s=elapsed, ok=False,
            note=f"could not parse output tail={tail!r}",
        )

    n_tokens = int(n_tokens_m.group(1))
    ids_str = ids_m.group(1).strip()
    ids = [int(x) for x in ids_str.split(",")] if ids_str else []
    first_token = ids[0] if ids else None
    # Second token is the first position actually reachable by the MTP
    # accept/reject path: position 0 (first_token) is emitted by the
    # chunked-prefill's own categorical sampler BEFORE the spec-decode loop
    # is even entered (see Generate.swift: "the seed token is already
    # known (picked by prefill's sampler)") — that draw is identical
    # between arms for a fixed seed regardless of BUTTER_QWEN_MTP_SPEC,
    # since it depends only on the (arm-independent) prefill logits + RNG
    # state at that exact point. It is therefore CONFOUNDED as a test of
    # the accept scheme specifically (expect near-tautological agreement
    # across arms) even though it's still useful as a first-token
    # gross-distribution sanity check. second_token is the first position
    # whose value can actually differ based on MTP draft/accept/reject
    # behavior, so it is the more targeted diagnostic.
    second_token = ids[1] if len(ids) > 1 else None
    digest = hashlib.sha256(ids_str.encode("utf-8")).hexdigest()[:16]

    return TrialResult(
        arm=arm, seed=seed, first_token=first_token, second_token=second_token,
        n_tokens=n_tokens, digest=digest, elapsed_s=elapsed, ok=True,
    )


# ─── Determinism control ───────────────────────────────────────────────────


def determinism_control(
    binary: Path, model: str, prompt: str, max_tokens: int,
    temperature: float, top_p: float, top_k: int, control_seed: int,
    watchdog_s: int,
) -> bool:
    """Per-arm: run the SAME seed twice, require identical digests.

    This is the precondition for everything downstream: the statistical
    tests treat each run as an independent draw conditioned on its seed.
    If reruns of the same seed don't reproduce, seeding isn't actually
    controlling the sampler and the whole harness is void (STOP per the
    task brief — this is not a soft warning).
    """
    print(f"=== Determinism control (seed={control_seed}) ===")
    all_ok = True
    for arm, arm_env in (("spec-on", SPEC_ON_ENV), ("spec-off", SPEC_OFF_ENV)):
        r1 = run_trial(
            binary, model, prompt, max_tokens, temperature, top_p, top_k,
            control_seed, arm, arm_env, watchdog_s,
        )
        r2 = run_trial(
            binary, model, prompt, max_tokens, temperature, top_p, top_k,
            control_seed, arm, arm_env, watchdog_s,
        )
        if not (r1.ok and r2.ok):
            print(f"  {arm}: FAIL — a control run did not complete "
                  f"(r1.ok={r1.ok} note={r1.note!r}, r2.ok={r2.ok} note={r2.note!r})")
            all_ok = False
            continue
        match = r1.digest == r2.digest
        print(
            f"  {arm}: run1 digest={r1.digest} run2 digest={r2.digest} "
            f"-> {'MATCH' if match else 'MISMATCH'}"
        )
        if not match:
            all_ok = False
    return all_ok


# ─── Statistics ─────────────────────────────────────────────────────────────


def chi2_sf_stdlib(x: float, df: int) -> float:
    """Chi-square survival function P(X > x) via the regularized upper
    incomplete gamma function Q(df/2, x/2), evaluated with the standard
    series (x < a+1) / continued-fraction (x >= a+1) split (Numerical
    Recipes "gammq"). Stdlib-only fallback for when scipy isn't available.
    """
    a = df / 2.0
    x = x / 2.0
    if x < 0 or a <= 0:
        return 1.0
    if x == 0:
        return 1.0
    if x < a + 1.0:
        # Series representation of the LOWER incomplete gamma P(a, x),
        # then Q = 1 - P.
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(500):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * 1e-14:
                break
        p = summ * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1.0 - p))
    else:
        # Continued fraction for the UPPER incomplete gamma Q(a, x)
        # directly (Lentz's algorithm).
        tiny = 1e-300
        b = x + 1.0 - a
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, 500):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-14:
                break
        q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
        return max(0.0, min(1.0, q))


def g_test_contingency(table: list[list[int]]) -> tuple[float, int, float]:
    """Likelihood-ratio chi-square (G-test) over an R x C contingency
    table. G = 2 * sum_i O_i * ln(O_i / E_i); df = (R-1)(C-1).
    Stdlib fallback when scipy is unavailable.
    """
    rows = len(table)
    cols = len(table[0])
    row_sums = [sum(r) for r in table]
    col_sums = [sum(table[r][c] for r in range(rows)) for c in range(cols)]
    total = sum(row_sums)
    g = 0.0
    for r in range(rows):
        for c in range(cols):
            o = table[r][c]
            if o == 0:
                continue
            e = row_sums[r] * col_sums[c] / total
            g += 2.0 * o * math.log(o / e)
    df = (rows - 1) * (cols - 1)
    p = chi2_sf_stdlib(g, df) if df > 0 else 1.0
    return g, df, p


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_whitney_u_stdlib(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test, normal approximation with tie
    correction. Stdlib fallback when scipy is unavailable.

        U1 = R1 - n1(n1+1)/2            (R1 = sum of ranks assigned to a)
        mu_U = n1 n2 / 2
        sigma_U = sqrt( n1 n2 / 12 * ((n1+n2+1) - tie_correction) )
        tie_correction = sum(t_i^3 - t_i) / ((n1+n2)(n1+n2-1))   [Sokal &
            Rohlf continuity correction for tied ranks; t_i = size of tie
            group i]
        z = (U1 - mu_U) / sigma_U
        p = 2 * (1 - Phi(|z|))
    """
    n1, n2 = len(a), len(b)
    combined = sorted((v, 0) for v in a) + sorted((v, 1) for v in b)
    combined.sort(key=lambda t: t[0])

    # Assign midranks for ties.
    ranks = [0.0] * len(combined)
    i = 0
    tie_term = 0.0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-indexed average rank over [i, j)
        t = j - i
        tie_term += t**3 - t
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    r1 = sum(rank for rank, (val, grp) in zip(ranks, combined) if grp == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    mu_u = n1 * n2 / 2.0
    if n <= 1:
        return u1, 1.0
    sigma_u_sq = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma_u_sq <= 0:
        # Degenerate: identical values in both groups (e.g. every run hit
        # the --max-tokens cap). No evidence of a difference.
        return u1, 1.0
    sigma_u = math.sqrt(sigma_u_sq)
    z = (u1 - mu_u) / sigma_u
    p = 2.0 * (1.0 - normal_cdf(abs(z)))
    return u1, max(0.0, min(1.0, p))


def token_category_test(spec_on: list[int], spec_off: list[int]) -> dict:
    """Contingency test on a token-id position (first or second token):
    categories = ids recurring >=2 times pooled across both arms, tail
    pooled into 'other'. Reused for both the required first-token stat and
    the supplementary second-token diagnostic.
    """
    from collections import Counter

    pooled = Counter(spec_on) + Counter(spec_off)
    categories = sorted([tok for tok, c in pooled.items() if c >= 2])
    cat_index = {tok: i for i, tok in enumerate(categories)}
    ncat = len(categories) + 1  # +1 for pooled "other"
    other_idx = ncat - 1

    def row(ids: list[int]) -> list[int]:
        r = [0] * ncat
        for t in ids:
            r[cat_index.get(t, other_idx)] += 1
        return r

    table = [row(spec_on), row(spec_off)]
    engine = "scipy.chi2_contingency"
    if HAVE_SCIPY:
        # Drop all-zero columns (chi2_contingency errors on them).
        cols_used = [c for c in range(ncat) if table[0][c] + table[1][c] > 0]
        trimmed = [[table[r][c] for c in cols_used] for r in range(2)]
        if len(cols_used) < 2:
            return {
                "engine": engine, "statistic": float("nan"), "df": 0,
                "p_value": 1.0, "n_categories": len(cols_used),
                "note": "degenerate: <2 non-empty categories, no evidence of difference",
            }
        stat, p, dof, _ = _scipy_stats.chi2_contingency(trimmed)
    else:
        engine = "stdlib G-test"
        stat, dof, p = g_test_contingency(table)

    return {
        "engine": engine, "statistic": stat, "df": dof, "p_value": p,
        "n_categories": ncat, "note": "",
    }


def length_test(spec_on: list[int], spec_off: list[int]) -> dict:
    if HAVE_SCIPY:
        try:
            res = _scipy_stats.mannwhitneyu(spec_on, spec_off, alternative="two-sided")
            return {
                "engine": "scipy.mannwhitneyu", "statistic": res.statistic,
                "p_value": res.pvalue,
            }
        except ValueError as e:
            # scipy raises on all-identical input (zero variance).
            return {"engine": "scipy.mannwhitneyu", "statistic": float("nan"),
                    "p_value": 1.0, "note": f"degenerate: {e}"}
    u, p = mann_whitney_u_stdlib([float(x) for x in spec_on], [float(x) for x in spec_off])
    return {"engine": "stdlib Mann-Whitney U", "statistic": u, "p_value": p}


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--prompt", type=str, default="The history of computing is")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--n", type=int, default=30, help="trials per arm")
    ap.add_argument("--base-seed", type=int, default=20260807)
    ap.add_argument("--control-seed", type=int, default=999)
    ap.add_argument("--watchdog", type=int, default=WATCHDOG_SECONDS)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--out", type=Path, required=True, help="TSV output path")
    args = ap.parse_args()

    if not args.binary.exists():
        print(f"ERROR: butter binary not found at {args.binary}", file=sys.stderr)
        return 2

    print(f"scipy available: {HAVE_SCIPY}")
    print(f"binary: {args.binary}")
    print(f"model: {args.model}")
    print(
        f"sampling config: temperature={args.temperature} top_p={args.top_p} "
        f"top_k={args.top_k} max_tokens={args.max_tokens} prompt={args.prompt!r}"
    )
    print(f"spec-on env: {SPEC_ON_ENV}")
    print(f"spec-off env: (spec vars unset)")

    ok0, msg0 = preflight_check()
    print(msg0)
    if not ok0:
        print("ERROR: preflight memory check failed before any run. Aborting.")
        return 3

    # ── Step 1: determinism control (STOP on failure) ──────────────────
    control_ok = determinism_control(
        args.binary, args.model, args.prompt, args.max_tokens,
        args.temperature, args.top_p, args.top_k, args.control_seed,
        args.watchdog,
    )
    if not control_ok:
        print(
            "\nSTOP: determinism control FAILED — same-seed reruns produced "
            "different digests (or a run failed outright). Seeding is not "
            "actually controlling the sampler, so the seeded-independent-"
            "draws assumption behind the statistical tests below does not "
            "hold. The harness is VOID until this is fixed. Not running the "
            "statistical sweep."
        )
        return 1
    print("Determinism control: PASS (both arms reproduce identically on same seed).\n")

    # ── Step 2: statistical sweep, N trials per arm, distinct seeds ────
    seeds = [args.base_seed + i for i in range(args.n)]
    rows: list[TrialResult] = []
    for arm, arm_env in (("spec-off", SPEC_OFF_ENV), ("spec-on", SPEC_ON_ENV)):
        print(f"=== Sweep: {arm} (N={args.n}) ===")
        for i, seed in enumerate(seeds):
            r = run_trial(
                args.binary, args.model, args.prompt, args.max_tokens,
                args.temperature, args.top_p, args.top_k, seed, arm, arm_env,
                args.watchdog,
            )
            rows.append(r)
            status = "ok" if r.ok else f"FAIL ({r.note})"
            print(
                f"  [{i + 1}/{args.n}] seed={seed} {status} "
                f"first_token={r.first_token} second_token={r.second_token} "
                f"n_tokens={r.n_tokens} digest={r.digest} elapsed={r.elapsed_s:.1f}s"
            )

    # ── Write raw TSV ────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["arm", "seed", "first_token", "second_token", "n_tokens",
                     "digest", "ok", "elapsed_s", "note"])
        for r in rows:
            w.writerow([r.arm, r.seed, r.first_token, r.second_token, r.n_tokens,
                        r.digest, r.ok, f"{r.elapsed_s:.2f}", r.note])
    print(f"\nWrote raw per-run TSV: {args.out}")

    # ── Statistics ───────────────────────────────────────────────────
    spec_on_ok = [r for r in rows if r.arm == "spec-on" and r.ok]
    spec_off_ok = [r for r in rows if r.arm == "spec-off" and r.ok]
    n_fail = sum(1 for r in rows if not r.ok)

    print("\n=== VERDICT ===")
    print(f"trials: N={args.n} per arm; spec-on ok={len(spec_on_ok)}/{args.n}, "
          f"spec-off ok={len(spec_off_ok)}/{args.n}; total failures={n_fail}")
    if n_fail:
        print("  (failed trials excluded from the statistics below — see TSV 'note' column)")

    if len(spec_on_ok) < 2 or len(spec_off_ok) < 2:
        print("UNDERPOWERED: fewer than 2 successful trials in an arm — no "
              "statistically valid conclusion. Report as underpowered, not "
              "pass/fail (RST-AI Loop E Failure handling).")
        return 1

    ft_res = token_category_test(
        [r.first_token for r in spec_on_ok], [r.first_token for r in spec_off_ok]
    )
    len_res = length_test(
        [r.n_tokens for r in spec_on_ok], [r.n_tokens for r in spec_off_ok]
    )

    # Structural check on the required stats, printed as findings rather
    # than silently trusted: (a) first-token is drawn by the chunked-
    # prefill's own categorical sampler BEFORE the spec-decode loop is
    # entered (Generate.swift), so for a fixed seed it depends only on
    # arm-independent prefill logits + RNG state — expect near-total
    # seed-paired agreement regardless of whether the accept scheme is
    # unbiased. (b) length is tokens-to-stop capped at --max-tokens; if
    # the prompt/cap combination never lets EOS fire, every run in both
    # arms hits the cap and the length test is measuring zero variance.
    paired_first_match = sum(
        1 for a, b in zip(spec_on_ok, spec_off_ok)
        if a.seed == b.seed and a.first_token == b.first_token
    )
    n_lens_on = len(set(r.n_tokens for r in spec_on_ok))
    n_lens_off = len(set(r.n_tokens for r in spec_off_ok))
    first_token_confounded = paired_first_match == len(spec_on_ok)
    length_degenerate = n_lens_on <= 1 and n_lens_off <= 1

    print(
        f"1. FIRST-TOKEN distribution ({ft_res['engine']}, "
        f"{ft_res['n_categories']} categories incl. pooled tail, "
        f"df={ft_res.get('df')}): "
        f"statistic={ft_res['statistic']:.4f}  p={ft_res['p_value']:.4g}"
        + (f"  [{ft_res['note']}]" if ft_res.get("note") else "")
    )
    if first_token_confounded:
        print(
            f"   FINDING: {paired_first_match}/{len(spec_on_ok)} seed-paired runs "
            "have IDENTICAL first_token across arms. This is expected — see "
            "comment above — and means this stat mainly verifies the "
            "prefill-sampler path is arm-invariant, NOT that the MTP "
            "accept/reject scheme is unbiased. Treat p-value here as weak "
            "evidence; see stat 1b (second-token) for a more targeted check."
        )
    print(
        f"2. LENGTH (tokens-to-stop) distribution ({len_res['engine']}): "
        f"statistic={len_res['statistic']:.4f}  p={len_res['p_value']:.4g}"
        + (f"  [{len_res['note']}]" if len_res.get("note") else "")
    )
    if length_degenerate:
        print(
            f"   FINDING: spec-on n_tokens values={sorted(set(r.n_tokens for r in spec_on_ok))}, "
            f"spec-off n_tokens values={sorted(set(r.n_tokens for r in spec_off_ok))} — "
            "every run in both arms hit --max-tokens without EOS firing. "
            "Zero variance in both arms means this Mann-Whitney result is "
            "VACUOUS (p=1 by construction, not by distributional agreement). "
            "Testability gap: this prompt/max-tokens config gives the length "
            "stat no power to detect anything. A prompt/cap combination that "
            "actually lets EOS fire before the cap (or shorter --max-tokens) "
            "is needed for this stat to carry information."
        )

    # 1b. Supplementary diagnostic (NOT part of the alpha=0.01 gate below,
    # which is scoped exactly to the two stats the task specified): second
    # token is the first position genuinely reachable by the spec-decode
    # accept/reject path, so it is the more targeted test of the actual
    # claim under test.
    on2 = [r.second_token for r in spec_on_ok if r.second_token is not None]
    off2 = [r.second_token for r in spec_off_ok if r.second_token is not None]
    st_res = None
    if len(on2) >= 2 and len(off2) >= 2:
        st_res = token_category_test(on2, off2)
        print(
            f"1b. [diagnostic, not gated] SECOND-TOKEN distribution "
            f"({st_res['engine']}, {st_res['n_categories']} categories, "
            f"df={st_res.get('df')}): statistic={st_res['statistic']:.4f} "
            f"p={st_res['p_value']:.4g}"
            + (f"  [{st_res['note']}]" if st_res.get("note") else "")
        )

    print(f"3. Determinism control: PASS (both arms; same-seed reruns identical digest)")

    alpha = args.alpha
    pass_ft = ft_res["p_value"] > alpha
    pass_len = len_res["p_value"] > alpha
    verdict = "PASS" if (pass_ft and pass_len) else "FAIL"

    print(f"\nalpha = {alpha}  (gate = stats 1 + 2 only, per task spec; stat 1b is diagnostic)")
    print(f"first-token test:  p={ft_res['p_value']:.4g}  {'> alpha (no evidence of difference)' if pass_ft else '<= alpha (SIGNIFICANT DIFFERENCE)'}")
    print(f"length test:       p={len_res['p_value']:.4g}  {'> alpha (no evidence of difference)' if pass_len else '<= alpha (SIGNIFICANT DIFFERENCE)'}")
    if st_res is not None:
        pass_st = st_res["p_value"] > alpha
        print(f"second-token test: p={st_res['p_value']:.4g}  {'> alpha (no evidence of difference)' if pass_st else '<= alpha (SIGNIFICANT DIFFERENCE)'}  [diagnostic]")
    print(f"\nVERDICT: {verdict} at alpha={alpha}")
    if verdict == "FAIL":
        print(
            "  Failing this gate means the sampled-MTP accept/match scheme "
            "measurably biases which tokens get emitted relative to the "
            "no-spec reference sampler — i.e. it is NOT a lossless "
            "rejection-sampling implementation. This is exactly the bug "
            "class rejection sampling exists to prevent (an accept rule "
            "that silently distorts the target distribution instead of "
            "only accelerating unbiased draws from it). Per RST-AI Loop E "
            "step 7: pull the specific failing trials from the TSV above "
            "and read them directly before concluding whether the drift is "
            "a systematic bug (clustered) or the test under-detecting "
            "noise (scattered, reconsider N)."
        )
    else:
        print(
            "  No evidence of a distributional difference between spec-on "
            "and spec-off at this alpha/N. Per RST-AI Loop E step 8: this "
            "is 'no detected regression at N={}', not a proof of "
            "losslessness — increase N if this becomes higher-stakes.".format(args.n)
        )

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
