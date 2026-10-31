"""Japan market readiness score (0-100, grade A-F) combining compliance, competitive position and label quality.
Weights: compliance 50, competitive position 30, label quality 20."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ReadinessComponent:
    """One scored section of the readiness report, with its grade, status line and bullet breakdown."""
    name:        str
    score:       int          # 0 to max_score
    max_score:   int
    grade:       str          # A / B / C / D / F
    status:      str          # one-line verdict
    breakdown:   List[str]    # bullet points shown under the component


@dataclass
class ReadinessReport:
    """Overall readiness score, grade and headline, plus its components and top actions."""
    overall_score:  int           # 0–100
    letter_grade:   str           # A / B / C / D / F
    headline:       str           # e.g. "Ready for Japanese market"
    sub_headline:   str           # e.g. "2 compliance issues need fixing"
    components:     List[ReadinessComponent]
    top_actions:    List[str]     # top 3 things to fix, for the header card


def _letter(score: int, max_score: int) -> str:
    """Letter grade for a component score, based on its percentage of max_score."""
    pct = score / max_score * 100 if max_score else 0
    if pct >= 90: return "A"
    if pct >= 75: return "B"
    if pct >= 60: return "C"
    if pct >= 45: return "D"
    return "F"


def _overall_letter(score: int) -> str:
    """Letter grade for the 0-100 overall score."""
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"


def _compliance_component(compliance_reports: Dict[str, Any]) -> ReadinessComponent:
    """Compliance score out of 50: -8 per error, -3 per warning, N/A checks ignored.
    Returns (component, top_actions)."""
    MAX = 50
    deductions = 0
    breakdown  = []
    top_actions = []

    for key, report in compliance_reports.items():
        if report.not_applicable:
            breakdown.append(f"· {report.standard} — N/A (not applicable to this product)")
            continue

        errors   = [i for i in report.issues if i.severity == "error"]
        warnings = [i for i in report.issues if i.severity == "warning"]
        n_pass   = len(report.passed_checks)

        if not errors and not warnings:
            breakdown.append(f"✓ {report.standard} — passed ({n_pass} checks)")
        else:
            for e in errors:
                deductions += 8
                breakdown.append(f"✗ {report.standard} [{e.code}] — {e.message[:80]}…")
                top_actions.append(f"Fix {e.code}: {e.message[:60]}…")
            for w in warnings:
                deductions += 3
                breakdown.append(f"⚠ {report.standard} [{w.code}] — {w.message[:80]}…")

    score = max(0, MAX - deductions)

    if score == MAX:
        status = "All compliance checks passed — no legal blockers"
    elif score >= 38:
        status = "Minor compliance issues — fixable before launch"
    elif score >= 25:
        status = "Moderate compliance gaps — label revision required"
    else:
        status = "Critical violations — product cannot be sold in Japan as-is"

    return ReadinessComponent(
        name="Compliance (法令適合性)",
        score=score,
        max_score=MAX,
        grade=_letter(score, MAX),
        status=status,
        breakdown=breakdown,
    ), top_actions


def _competitive_component(bench_report: Dict[str, Any]) -> ReadinessComponent:
    """Competitive score out of 30 from the benchmark percentile (>=70: 30, >=50: 22, >=35: 15, else 8).
    Falls back to a neutral 15 if benchmarking is unavailable."""
    MAX = 30

    if "error" in bench_report or not bench_report:
        return ReadinessComponent(
            name="Competitive Position (競合比較)",
            score=15,
            max_score=MAX,
            grade="C",
            status="Benchmarking unavailable — score estimated at midpoint",
            breakdown=["· Could not compare against category peers"],
        ), []

    pct        = bench_report.get("percentile", 50)
    strengths  = bench_report.get("strengths",  [])
    weaknesses = bench_report.get("weaknesses", [])
    category   = bench_report.get("category_label", "Unknown category")
    n_better   = bench_report.get("better_than_avg_count", 0)
    n_total    = bench_report.get("total_metrics", 1)
    top_actions = []

    if pct >= 70:
        score = 30
    elif pct >= 50:
        score = 22
    elif pct >= 35:
        score = 15
    else:
        score = 8

    breakdown = [
        f"· Category: {category}",
        f"· Better than average on {n_better}/{n_total} metrics ({pct}% score)",
    ]
    if strengths:
        breakdown.append(f"✓ Strengths: {', '.join(strengths)}")
    if weaknesses:
        breakdown.append(f"⚠ Behind avg: {', '.join(weaknesses)}")
        for w in weaknesses[:2]:
            top_actions.append(f"Improve {w} to match category average")

    if score >= 25:
        status = f"Strong market position — better than {pct}% of {category} products"
    elif score >= 15:
        status = f"Competitive — room to improve on {len(weaknesses)} metric(s)"
    else:
        status = f"Below average — significant reformulation recommended"

    return ReadinessComponent(
        name="Competitive Position (競合比較)",
        score=score,
        max_score=MAX,
        grade=_letter(score, MAX),
        status=status,
        breakdown=breakdown,
    ), top_actions


def _label_quality_component(suggestions) -> ReadinessComponent:
    """Label quality score out of 20: -5 per high and -2 per medium suggestion, low ones are free.
    Compliance suggestions are skipped since compliance already counts them."""
    MAX = 20
    deductions = 0
    breakdown  = []
    top_actions = []

    non_compliance = [s for s in suggestions if s.category != "Compliance"]
    compliance_sug = [s for s in suggestions if s.category == "Compliance"]

    for s in non_compliance:
        if s.priority == "high":
            deductions += 5
            breakdown.append(f"✗ [HIGH] {s.title}")
            top_actions.append(s.title)
        elif s.priority == "medium":
            deductions += 2
            breakdown.append(f"⚠ [MED]  {s.title}")
        else:
            breakdown.append(f"· [LOW]  {s.title}")

    if compliance_sug:
        breakdown.append(
            f"· {len(compliance_sug)} compliance suggestion(s) counted in Component 1"
        )

    if not non_compliance:
        breakdown.append("✓ No reformulation or positioning issues detected")

    score = max(0, MAX - deductions)

    if score == MAX:
        status = "Label and product profile — no improvements needed"
    elif score >= 15:
        status = f"{len(non_compliance)} minor improvement(s) identified"
    elif score >= 10:
        status = f"{len(non_compliance)} suggestions — medium-effort improvements available"
    else:
        status = f"Multiple high-priority improvements needed before launch"

    return ReadinessComponent(
        name="Label & Product Quality (品質・表示)",
        score=score,
        max_score=MAX,
        grade=_letter(score, MAX),
        status=status,
        breakdown=breakdown,
    ), top_actions


def compute_readiness(
    compliance_reports: Dict[str, Any],
    bench_report: Dict[str, Any],
    suggestions: list,
    bench_confirmed: bool = False,
) -> ReadinessReport:
    """Compute the Japan market readiness report from compliance, benchmark and suggestion outputs.
    Until bench_confirmed is True (user ran the benchmark), competitive position stays at a neutral 15/30."""
    comp1, actions1 = _compliance_component(compliance_reports)
    comp2, actions2 = _competitive_component(bench_report) if bench_confirmed else (
        ReadinessComponent(
            name="Competitive Position (競合比較)",
            score=15,
            max_score=30,
            grade="C",
            status="⏳ Run the benchmark in the Competitive Benchmarking tab to score this section",
            breakdown=["· Select a category card and confirm label basis to get your competitive score"],
        ), []
    )
    comp3, actions3 = _label_quality_component(suggestions)

    overall = comp1.score + comp2.score + comp3.score

    grade = _overall_letter(overall)

    # compliance actions first, then competitive, then label
    all_actions = actions1 + actions2 + actions3
    top_actions = all_actions[:3] if all_actions else ["No critical actions required"]

    headlines = {
        "A": ("Ready for Japanese Market", "Product meets legal and market standards"),
        "B": ("Nearly Market-Ready", "Minor fixes needed before launch"),
        "C": ("Moderate Work Required", "Label or compliance revision recommended"),
        "D": ("Significant Barriers", "Legal or market issues must be resolved"),
        "F": ("Not Market-Ready", "Critical violations — cannot legally sell in Japan"),
    }
    headline, sub_headline = headlines[grade]

    n_errors = sum(
        1 for r in compliance_reports.values()
        for i in r.issues if i.severity == "error"
        and not r.not_applicable
    )
    if n_errors > 0:
        sub_headline = (
            f"{n_errors} legal violation(s) must be fixed before market entry"
        )
    elif comp1.score == 50 and comp2.score >= 22:
        sub_headline = "Compliance clear — focus on competitive positioning"

    return ReadinessReport(
        overall_score=overall,
        letter_grade=grade,
        headline=headline,
        sub_headline=sub_headline,
        components=[comp1, comp2, comp3],
        top_actions=top_actions,
    )