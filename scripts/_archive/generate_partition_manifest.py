#!/usr/bin/env python3
"""
Content-based partition manifest generator.

Reads every vault file's full text (frontmatter + body) and scores it
against domain vocabularies to classify into:
  - neuroscience  → vault-neuroscience
  - finance       → vault-finance
  - personal      → vault-personal
  - main          → stays in main vault (AI/software/infrastructure)

Outputs:
  data/partition-manifest.json       — {rel_path: target_vault} for files to migrate
  data/partition-review.json         — ambiguous files needing manual review
  data/partition-scores-debug.json   — full per-file scores for debugging

Usage:
  uv run python scripts/generate_partition_manifest.py [--dry-run] [--min-score 0.004]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import frontmatter as fm_lib

PROJECT_ROOT = Path(__file__).parent.parent
VAULT_PATH = Path("/mnt/external/obsidian-vault")
DATA_DIR = PROJECT_ROOT / "data"

# Directories to skip entirely
SKIP_DIRS = {"inbox", "wiki", "_archived", "_templates", "_bases", ".obsidian",
             "ai-dialogue", "account", "asset", "config"}

# Directories to never migrate (always stay in main)
STAY_DIRS = {"wiki", "ai-dialogue", "account", "asset", "config", "_archived"}

# ── Domain vocabularies ────────────────────────────────────────────────────────
# Terms are lowercased. Weight > 1.0 = strong domain signal.

NEURO_TERMS: dict[str, float] = {
    # Hard anatomy / physiology
    "neuroscience": 3.0, "neuroanatomy": 3.0, "neurology": 3.0,
    "neurophysiology": 3.0, "neuropharmacology": 3.0, "neurochemistry": 3.0,
    "neuroimaging": 2.5, "electrophysiology": 2.5, "neurodegeneration": 2.5,
    "neuronal": 2.0, "neuron": 2.0, "axon": 2.0, "dendrite": 2.0,
    "synapse": 2.0, "synaptic": 2.0, "myelin": 2.0, "myelination": 2.0,
    "glia": 2.0, "glial": 2.0, "astrocyte": 2.0, "oligodendrocyte": 2.0,
    "microglia": 2.0, "schwann": 2.0,
    # Brain regions
    "hippocampus": 2.5, "amygdala": 2.5, "cortex": 2.0, "cerebellum": 2.5,
    "brainstem": 2.5, "thalamus": 2.5, "hypothalamus": 2.5, "basal ganglia": 2.5,
    "substantia nigra": 3.0, "locus coeruleus": 3.0, "prefrontal": 2.0,
    "frontal lobe": 2.0, "temporal lobe": 2.0, "parietal lobe": 2.0,
    "occipital": 2.0, "limbic": 2.0, "striatum": 2.5, "putamen": 2.5,
    "caudate": 2.5, "nucleus accumbens": 2.5, "vmPFC": 2.0, "dlPFC": 2.0,
    "spinal cord": 2.5, "dorsal column": 3.0, "spinothalamic": 3.0,
    "blood-brain barrier": 2.5, "BBB": 2.0, "circumventricular": 3.0,
    # Physiology
    "action potential": 2.5, "membrane potential": 2.5, "resting potential": 2.5,
    "depolarization": 2.5, "hyperpolarization": 2.5, "LTP": 2.5, "LTD": 2.5,
    "synaptic plasticity": 3.0, "long-term potentiation": 3.0,
    "ion channel": 2.5, "voltage-gated": 2.5, "GABA": 2.5, "glutamate": 2.5,
    "NMDA": 2.5, "AMPA": 2.5, "dopamine": 2.0, "serotonin": 2.0,
    "norepinephrine": 2.0, "acetylcholine": 2.0, "receptor": 1.2,
    # Clinical
    "brown-séquard": 3.0, "UMN": 2.5, "LMN": 2.5, "upper motor neuron": 3.0,
    "lower motor neuron": 3.0, "spasticity": 2.5, "hyperreflexia": 2.5,
    "areflexia": 2.5, "parkinson": 2.5, "alzheimer": 2.5, "multiple sclerosis": 2.5,
    "neurological": 2.0, "cranial nerve": 2.5, "reflex arc": 2.5,
    "syringomyelia": 3.0, "demyelination": 2.5, "neuropathy": 2.5,
    "clinical reasoning": 2.5, "clinical neuroscience": 3.0, "diagnosis": 1.0,
    # Pharmacology
    "pharmacology": 2.5, "pharmacokinetics": 2.5, "pharmacodynamics": 2.5,
    "blood brain barrier": 2.5, "drug": 1.0, "medication": 1.0,
    "therapeutics": 2.0, "agonist": 2.0, "antagonist": 2.0,
    # Cognitive science
    "predictive processing": 3.0, "free energy principle": 3.0,
    "cognitive neuroscience": 3.0, "cognitive science": 2.5,
    "perception": 1.5, "attention": 1.5, "metacognition": 2.0,
    "consciousness": 2.0, "working memory": 2.0, "executive function": 2.0,
    "psycholinguistics": 3.0, "cognitive psychology": 2.5,
    # Research / academic
    "fMRI": 2.5, "EEG": 2.5, "MEG": 2.5, "TMS": 2.0,
    "TRIBE v2": 3.0, "brain encoding": 3.0, "neural encoding": 2.5,
    "NEURO ": 3.0, "PSYCH ": 2.5,  # course codes
    "neuroanatomy notes": 3.0, "neuro 302": 3.0, "neuro 404": 3.0,
    "psych 490": 3.0, "psych 491": 3.0,
    "histology": 2.5, "neurochemist": 3.0,
}

FINANCE_TERMS: dict[str, float] = {
    # Quantitative finance — specific compound terms only (avoid single-word false positives)
    "quantitative finance": 3.0, "behavioral finance": 3.0,
    "econophysics": 3.0, "market microstructure": 3.0,
    "quant finance": 3.0, "quant research": 3.0,
    "sharpe ratio": 3.0, "sortino ratio": 3.0, "sharpe": 2.5, "sortino": 2.5,
    "drawdown": 2.5, "annualized return": 2.5, "risk-adjusted return": 2.5,
    "backtest": 2.5, "backtesting": 2.5, "momentum strategy": 2.5,
    "mean reversion": 2.5, "factor model": 2.5, "risk management": 1.5,
    "alpha generation": 3.0, "portfolio alpha": 2.5,
    # Markets / instruments — specific enough to not false-positive
    "trading strategy": 2.5, "financial markets": 2.5, "stock market": 2.5,
    "investing": 2.0, "investment strategy": 2.5,
    "portfolio management": 2.5, "asset allocation": 2.5,
    "options trading": 2.5, "futures contract": 2.5, "derivatives": 2.0,
    "ETF": 2.5, "DCA": 2.5, "dollar-cost averaging": 3.0,
    "sector rotation": 2.5, "equity market": 2.5,
    "technical analysis": 2.5, "fundamental analysis": 2.5,
    "price action": 2.0, "market liquidity": 2.5, "order flow": 2.5,
    "bid-ask spread": 3.0, "market maker": 2.5,
    # Crypto — specific
    "cryptocurrency": 2.5, "bitcoin": 2.5, "ethereum": 2.5,
    "memecoin": 3.0, "crypto market": 2.5, "defi": 2.5,
    "FOMO": 2.5, "pump and dump": 2.5, "crypto volatility": 3.0,
    # Behavioral finance
    "prediction market": 3.0, "prediction markets": 3.0, "polymarket": 3.0,
    "behavioral survey": 3.0, "T1 traders": 3.0, "insider information": 2.5,
    "anchoring bias": 2.5, "herding behavior": 2.5, "overconfidence bias": 2.5,
    "loss aversion": 2.5, "prospect theory": 2.5,
    "behavioral-quantitative": 3.0, "market sentiment": 2.5,
    # Research methods — finance-specific compound forms only
    "statistical arbitrage": 3.0, "cronbach alpha": 3.0, "likert scale": 2.5,
    "survey design": 2.0, "research methodology": 1.5,
    # Tools
    "yfinance": 3.0, "vectorbt": 3.0, "backtrader": 3.0, "zipline": 3.0,
    "sector flow": 3.0, "memecoin sentiment": 3.0,
    # Finance course/learning
    "econophysics course": 3.0, "finance learning roadmap": 3.0,
    "quantitative finance learning": 3.0,
}

PERSONAL_TERMS: dict[str, float] = {
    # Career / professional (personal, not technical)
    "resume": 2.5, "job search": 3.0, "job application": 3.0,
    "career narrative": 3.0, "personal branding": 3.0, "linkedin": 2.5,
    "career strategy": 2.5, "career transition": 2.5, "portfolio narrative": 3.0,
    # Identity / self — specific compound forms to avoid generic false positives
    "self-awareness": 3.0, "self-regulation": 3.0, "self-mastery": 3.0,
    "personal development": 2.5, "self-improvement": 2.5,
    "personal journal": 3.0, "journal entry": 3.0, "personal reflection": 2.5,
    "authenticity": 2.5, "vulnerability": 2.5, "personal identity": 2.5,
    # Lifestyle / health (non-clinical) — highly specific terms
    "lucid dreaming": 3.0, "lucid dream": 3.0, "dream journal": 3.0,
    "sleep optimization": 2.5, "sleep environment": 2.5,
    "peptide protocol": 3.0, "BPC-157": 3.0, "TB-500": 3.0, "GH peptide": 3.0,
    "CJC": 3.0, "bacteriostatic water": 3.0, "subcutaneous injection": 2.5,
    "cannabis use": 2.5, "nootropic": 2.5,
    "cooking": 3.0, "meal prep": 2.5, "recipe": 2.0,
    "workout routine": 2.5, "exercise routine": 2.5,
    # Relationships / social — specific
    "personal relationships": 2.5, "friendship": 2.5, "social skills": 2.5,
    "soft skills": 2.5, "emotional intelligence": 2.5,
    # Epistemics / philosophy
    "epistemics": 3.0, "epistemology": 2.5, "philosophy of mind": 2.5,
    "worldview": 2.0, "personal philosophy": 2.5,
    # Education (personal context — not class notes which go to neuro)
    "phi sigma rho": 3.0, "sorority": 3.0, "greek life": 3.0, "rush week": 3.0,
    "homework assignment": 2.5,
    # Gaming / hobbies — specific game names
    "gamescope": 3.0, "apex legends": 3.0, "thinkorswim": 2.5,
    "steam game": 2.5, "gaming setup": 2.5,
    # Personal finance (distinct from quant finance)
    "budgeting": 2.5, "personal finance": 3.0, "personal expenses": 2.5,
}

# Terms that strongly indicate content should STAY in main vault
MAIN_TERMS: dict[str, float] = {
    "alfred": 2.0, "alfred-v2": 3.0, "alfred daemon": 3.0,
    "milvus": 3.0, "vault curator": 3.0, "distiller": 3.0, "consolidator": 3.0,
    "surveyor": 3.0, "janitor": 2.5, "BM25": 3.0, "embeddings": 2.0,
    "ollama": 2.5, "nomic-embed": 3.0, "FastMCP": 3.0,
    "claude code": 2.5, "anthropic": 2.0, "claude": 1.5,
    "obsidian vault": 2.5, "vault_ops": 3.0, "session hook": 3.0,
    "emm": 2.0, "executive mind matrix": 3.0, "notion": 2.0,
    "canvas autopilot": 3.0, "canvas lms": 3.0,
    "crm": 1.5, "nova crm": 3.0, "crm-agentic": 3.0,
    "digital twin": 3.0, "digital-twin": 3.0,
    "hyprland": 3.0, "waybar": 3.0, "omarchy": 3.0,
    "docker": 2.0, "kubernetes": 2.5, "systemd": 2.5,
    "python": 1.0, "fastapi": 2.5, "nextjs": 2.5, "react": 1.5,
    "postgresql": 2.5, "redis": 2.5, "mongodb": 2.5,
    "authentication": 1.5, "JWT": 2.5, "OAuth": 2.5,
    "machine learning": 1.5, "deep learning": 2.0, "pytorch": 2.5,
    "LLM": 2.0, "language model": 2.0, "RAG": 2.5,
    "agent architecture": 2.5, "agentic": 2.0, "multi-agent": 2.5,
    "tribe-social": 3.0, "tribe v2": 3.0, "runpod": 3.0,
    "resume pipeline": 3.0, "adaptive resume": 3.0,
    "git": 1.5, "github": 1.5, "CI/CD": 2.5,
    "linux": 1.5, "arch linux": 2.5, "pacman": 2.5,
}


# Filename keyword → domain boosts (applied after vocab scoring)
# Keys are lowercase substrings matched against the file stem
FILENAME_BOOSTS: list[tuple[str, str, float]] = [
    # Neuroscience
    ("neuro", "neuroscience", 8.0), ("neurology", "neuroscience", 8.0),
    ("neuroanatomy", "neuroscience", 8.0), ("neurophysio", "neuroscience", 8.0),
    ("neuropharm", "neuroscience", 8.0), ("synap", "neuroscience", 6.0),
    ("axon", "neuroscience", 6.0), ("cortex", "neuroscience", 6.0),
    ("brainstem", "neuroscience", 8.0), ("spinal", "neuroscience", 6.0),
    ("pharmacol", "neuroscience", 6.0), ("hippocampus", "neuroscience", 8.0),
    ("clinical-neuroscience", "neuroscience", 10.0),
    ("cognitive-neuroscience", "neuroscience", 10.0),
    ("cognitive-science", "neuroscience", 8.0),
    ("cognitive-attention", "neuroscience", 8.0),
    ("predictive-processing", "neuroscience", 10.0),
    ("consciousness", "neuroscience", 6.0),
    ("psych-490", "neuroscience", 8.0), ("psych-491", "neuroscience", 8.0),
    ("neuro-302", "neuroscience", 8.0), ("neuro-404", "neuroscience", 8.0),
    ("tribe-v2", "neuroscience", 8.0), ("tribe v2", "neuroscience", 8.0),
    ("brain-encoding", "neuroscience", 8.0),
    # Finance
    ("behavioral-finance", "finance", 10.0), ("quantitative-finance", "finance", 10.0),
    ("econophysics", "finance", 10.0), ("prediction-market", "finance", 10.0),
    ("financial-market", "finance", 10.0), ("financial-behavior", "finance", 10.0),
    ("financial-data", "finance", 10.0), ("financial-research", "finance", 10.0),
    ("financial-learning", "finance", 10.0), ("financial-success", "finance", 10.0),
    ("market-mechanics", "finance", 10.0), ("market-microstructure", "finance", 10.0),
    ("memecoin", "finance", 10.0), ("crypto-market", "finance", 10.0),
    ("sector-flow", "finance", 10.0), ("trading", "finance", 6.0),
    ("backtesting", "finance", 8.0), ("portfolio-strategy", "finance", 8.0),
    ("price-reversal", "finance", 8.0),
    # Personal
    ("personal-journal", "personal", 10.0), ("lucid-dream", "personal", 10.0),
    ("career-narrative", "personal", 10.0), ("personal-branding", "personal", 10.0),
    ("resume", "personal", 8.0), ("job-search", "personal", 10.0),
    ("phi-sigma-rho", "personal", 10.0), ("peptide", "personal", 8.0),
    ("gaming", "personal", 6.0), ("gamescope", "personal", 8.0),
    ("self-awareness", "personal", 10.0), ("self-mastery", "personal", 10.0),
    ("personal-health", "personal", 8.0), ("personal-identity", "personal", 8.0),
    ("epistemics", "personal", 8.0), ("epistemology", "personal", 8.0),
]


def score_text(text: str, rel_path: str = "") -> dict[str, float]:
    """Score text against all 4 domain vocabularies. Returns normalized scores."""
    lower = text.lower()

    def _score(vocab: dict[str, float]) -> float:
        total = 0.0
        for term, weight in vocab.items():
            count = lower.count(term.lower())
            total += count * weight
        # Normalize by document length to avoid length bias
        words = max(len(lower.split()), 1)
        return total / (words ** 0.5)  # square-root normalization

    scores = {
        "neuroscience": _score(NEURO_TERMS),
        "finance": _score(FINANCE_TERMS),
        "personal": _score(PERSONAL_TERMS),
        "main": _score(MAIN_TERMS),
    }

    # Apply filename boosts — stem + parent dir both checked
    if rel_path:
        stem_lower = Path(rel_path).stem.lower()
        dir_lower = str(Path(rel_path).parent).lower()
        path_lower = rel_path.lower()
        for keyword, domain, boost in FILENAME_BOOSTS:
            if keyword in path_lower:
                scores[domain] = scores.get(domain, 0.0) + boost

    return scores


def classify(scores: dict[str, float], min_score: float) -> tuple[str, str]:
    """
    Returns (target, confidence) where target is one of:
    neuroscience / finance / personal / main / ambiguous
    confidence is: high / medium / low

    Ratio is computed between the top two TARGET domains only — main never
    suppresses a clear target domain signal just by being a noisy second place.
    Main wins only when it outscores all target domains outright.
    """
    main_score = scores.get("main", 0.0)

    target_domains = ["neuroscience", "finance", "personal"]
    target_ranked = sorted(
        [(d, scores.get(d, 0.0)) for d in target_domains],
        key=lambda x: x[1], reverse=True,
    )
    top_target, top_score = target_ranked[0]
    _second_target, second_score = target_ranked[1]

    # No target signal → stay in main
    if top_score < min_score:
        return "main", "high"

    # Main beats every target domain → stay in main
    if main_score >= top_score:
        return "main", "high"

    # Measure how much the top target beats the second target
    ratio = top_score / max(second_score, 1e-9)
    if ratio >= 2.5:
        confidence = "high"
    elif ratio >= 1.6:
        confidence = "medium"
    else:
        return "ambiguous", "low"

    return top_target, confidence


def extract_text(fp: Path) -> str:
    """Extract searchable text from a vault file."""
    try:
        post = fm_lib.load(str(fp))
        fm = post.metadata
        body = post.content or ""

        # Build text blob from frontmatter signals + body
        parts = [body]

        # Add frontmatter fields as weighted text
        for key in ("name", "subject", "tags", "type"):
            val = fm.get(key)
            if val:
                # Repeat field value to boost its weight
                parts.append(str(val) * 3)

        # cluster_sources are strong domain signals — repeat them
        sources = fm.get("cluster_sources", [])
        if sources:
            src_text = " ".join(str(s) for s in sources)
            parts.append(src_text * 5)  # high weight

        return " ".join(parts)
    except Exception:
        try:
            return fp.read_text(errors="replace")
        except Exception:
            return ""


def scan_files() -> list[tuple[str, Path]]:
    """Return (rel_path, abs_path) for all candidate vault files."""
    results = []
    for fp in sorted(VAULT_PATH.rglob("*.md")):
        rel = fp.relative_to(VAULT_PATH)
        parts = rel.parts

        # Skip hidden/system dirs
        if any(p.startswith(".") or p in SKIP_DIRS for p in parts):
            continue

        results.append((str(rel), fp))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-score", type=float, default=0.004,
                        help="Minimum domain score to classify (default 0.004)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing output files")
    parser.add_argument("--debug", action="store_true",
                        help="Write full per-file scores to partition-scores-debug.json")
    args = parser.parse_args()

    print("Content-based partition manifest generator")
    print(f"Vault: {VAULT_PATH}")
    print(f"Min score: {args.min_score}\n")

    files = scan_files()
    print(f"Scanning {len(files)} files...\n")

    manifest: dict[str, str] = {}    # rel_path → target vault
    review: list[dict] = []           # ambiguous files
    debug_scores: dict[str, dict] = {}
    stats: dict[str, Counter] = {
        "neuroscience": Counter(),
        "finance": Counter(),
        "personal": Counter(),
        "main": Counter(),
        "ambiguous": Counter(),
    }

    for rel_path, fp in files:
        text = extract_text(fp)
        scores = score_text(text, rel_path)
        target, confidence = classify(scores, args.min_score)

        dir_name = rel_path.split("/")[0]

        # Files in STAY_DIRS always stay in main
        if dir_name in STAY_DIRS:
            target = "main"
            confidence = "high"

        if args.debug:
            debug_scores[rel_path] = {
                "target": target,
                "confidence": confidence,
                "scores": {k: round(v, 5) for k, v in scores.items()},
            }

        if target in ("neuroscience", "finance", "personal"):
            manifest[rel_path] = target
            stats[target][confidence] += 1
        elif target == "ambiguous":
            review.append({
                "path": rel_path,
                "scores": {k: round(v, 5) for k, v in scores.items()},
            })
            stats["ambiguous"]["total"] += 1
        else:
            stats["main"][confidence] += 1

    # Print summary
    total_migrate = sum(sum(v.values()) for k, v in stats.items() if k not in ("main", "ambiguous"))
    print("Classification summary:")
    print(f"  → vault-neuroscience : {sum(stats['neuroscience'].values()):4d} files")
    print(f"  → vault-finance      : {sum(stats['finance'].values()):4d} files")
    print(f"  → vault-personal     : {sum(stats['personal'].values()):4d} files")
    print(f"  stays in main        : {sum(stats['main'].values()):4d} files")
    print(f"  ambiguous (review)   : {sum(stats['ambiguous'].values()):4d} files")
    print(f"\n  Total to migrate     : {total_migrate}")
    print(f"  Total files scanned  : {len(files)}")

    if not args.dry_run:
        DATA_DIR.mkdir(exist_ok=True)

        manifest_path = DATA_DIR / "partition-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"\nWrote: {manifest_path}")

        review_path = DATA_DIR / "partition-review.json"
        review_sorted = sorted(review, key=lambda x: max(x["scores"].values()), reverse=True)
        review_path.write_text(json.dumps(review_sorted, indent=2))
        print(f"Wrote: {review_path} ({len(review)} files to review)")

        if args.debug:
            debug_path = DATA_DIR / "partition-scores-debug.json"
            debug_path.write_text(json.dumps(debug_scores, indent=2, sort_keys=True))
            print(f"Wrote: {debug_path}")
    else:
        print("\n[dry-run] No files written.")
        if review:
            print(f"\nTop 10 ambiguous files:")
            for item in sorted(review, key=lambda x: max(x["scores"].values()), reverse=True)[:10]:
                top = sorted(item["scores"].items(), key=lambda x: x[1], reverse=True)[:2]
                print(f"  {item['path']}")
                print(f"    {top[0][0]}={top[0][1]:.4f}  {top[1][0]}={top[1][1]:.4f}")


if __name__ == "__main__":
    main()
