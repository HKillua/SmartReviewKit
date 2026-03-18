from __future__ import annotations


def render_report(report: dict) -> str:
    lines = [
        "=== Agent Eval Report ===",
        f"总用例: {report['total_cases']} | 通过: {report['passed_cases']} | 失败: {report['failed_cases']}",
        f"工具召回率:  {report['tool_recall_rate']:.1%}",
        f"工具精确率:  {report['tool_precision_rate']:.1%}",
        f"路径合理性:  {report['path_reasonableness_rate']:.1%}",
        f"输出覆盖度:  {report['output_coverage_rate']:.1%}",
    ]

    if report.get("proactive_cases"):
        lines.append(f"主动推荐命中率:  {report['proactive_hit_rate']:.1%}")
    if report.get("replan_cases"):
        lines.append(f"Replan 命中率:  {report['replan_hit_rate']:.1%}")
    if report.get("pacing_cases"):
        lines.append(f"Pacing 命中率:  {report['pacing_hit_rate']:.1%}")

    if report.get("failed_results"):
        lines.append("")
        for result in report["failed_results"]:
            lines.append(f"❌ {result['name']}")
            for failure in result.get("failures", []):
                lines.append(f"   - {failure}")

    return "\n".join(lines)
