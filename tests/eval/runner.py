from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from tests.eval.conftest import build_agent_for_case, load_eval_case
from tests.eval.report import render_report


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class EvalRunner:
    async def run_case(self, case):
        agent = build_agent_for_case(case)
        conversation_id = f"eval_{case.path.stem}"
        events = [event async for event in agent.chat(case.user_message, case.user_id, conversation_id)]
        await agent.flush()
        final_text = "".join(event.content or "" for event in events if event.type.value == "text_delta")
        done_event = next(event for event in events if event.type.value == "done")
        metadata = dict(done_event.metadata)
        event_tool_path = [
            str(event.tool_name)
            for event in events
            if event.type.value == "tool_start" and str(event.tool_name or "")
        ]
        tool_path = [str(value) for value in metadata.get("tool_path", []) if str(value)]
        if not tool_path:
            tool_path = event_tool_path
        control_mode = str(
            metadata.get("effective_control_mode")
            or metadata.get("planner_final_control_mode")
            or metadata.get("planner_control_mode")
            or ""
        )
        step_count = len(tool_path)
        expectations = case.expectations

        failures: list[str] = []

        for tool_name in expectations.get("tool_must_include", []) or []:
            if tool_name not in tool_path:
                failures.append(f"缺少必需工具: {tool_name}")

        for tool_name in expectations.get("tool_must_not_include", []) or []:
            if tool_name in tool_path:
                failures.append(f"出现了不允许的工具: {tool_name}")

        expected_mode = str(expectations.get("control_mode") or "").strip().lower()
        if expected_mode and control_mode.lower() != expected_mode:
            failures.append(f"control_mode 期望 {expected_mode}，实际 {control_mode or 'unknown'}")

        min_steps = expectations.get("min_steps")
        max_steps = expectations.get("max_steps")
        if min_steps is not None and step_count < int(min_steps):
            failures.append(f"步骤数过少，期望 >= {min_steps}，实际 {step_count}")
        if max_steps is not None and step_count > int(max_steps):
            failures.append(f"步骤数过多，期望 <= {max_steps}，实际 {step_count}")

        for needle in expectations.get("output_must_contain", []) or []:
            if str(needle) not in final_text:
                failures.append(f"输出缺少关键内容: {needle}")

        if "proactive_triggered" in expectations:
            expected = bool(expectations["proactive_triggered"])
            actual = bool(metadata.get("proactive_triggered", False))
            if actual != expected:
                failures.append(f"proactive_triggered 期望 {expected}，实际 {actual}")

        if "replan_triggered" in expectations:
            expected = bool(expectations["replan_triggered"])
            actual = bool(metadata.get("replan_triggered", False))
            if actual != expected:
                failures.append(f"replan_triggered 期望 {expected}，实际 {actual}")

        if "pacing_level" in expectations:
            expected = str(expectations["pacing_level"])
            actual = str(metadata.get("pacing_level", ""))
            if actual != expected:
                failures.append(f"pacing_level 期望 {expected}，实际 {actual or 'unknown'}")

        if "batch_evaluation" in expectations:
            expected = bool(expectations["batch_evaluation"])
            actual = bool(metadata.get("batch_evaluation", False))
            if actual != expected:
                failures.append(f"batch_evaluation 期望 {expected}，实际 {actual}")

        if "alignment_status" in expectations:
            expected = str(expectations["alignment_status"])
            actual = str(metadata.get("alignment_status", ""))
            if actual != expected:
                failures.append(f"alignment_status 期望 {expected}，实际 {actual or 'unknown'}")

        if "question_count" in expectations:
            expected = int(expectations["question_count"])
            actual = int(metadata.get("question_count", 0) or 0)
            if actual != expected:
                failures.append(f"question_count 期望 {expected}，实际 {actual}")

        return {
            "name": case.name,
            "passed": not failures,
            "failures": failures,
            "tool_path": tool_path,
            "control_mode": control_mode,
            "step_count": step_count,
            "final_text": final_text,
            "metadata": metadata,
            "expectations": expectations,
        }

    async def run_all(self, cases_dir: str) -> dict:
        case_paths = sorted(Path(cases_dir).glob("*.yaml"))
        cases = [load_eval_case(path) for path in case_paths]
        results = [await self.run_case(case) for case in cases]

        total_must_include = sum(len(result["expectations"].get("tool_must_include", []) or []) for result in results)
        hit_must_include = 0
        must_not_violations = 0
        reasonable_paths = 0
        output_hits = 0
        proactive_cases = 0
        proactive_hits = 0
        replan_cases = 0
        replan_hits = 0
        pacing_cases = 0
        pacing_hits = 0

        for result in results:
            expectations = result["expectations"]
            tool_path = set(result["tool_path"])
            for tool_name in expectations.get("tool_must_include", []) or []:
                if tool_name in tool_path:
                    hit_must_include += 1
            for tool_name in expectations.get("tool_must_not_include", []) or []:
                if tool_name in tool_path:
                    must_not_violations += 1

            min_steps = expectations.get("min_steps", 0)
            max_steps = expectations.get("max_steps", 999)
            if int(min_steps) <= result["step_count"] <= int(max_steps):
                reasonable_paths += 1

            required_terms = expectations.get("output_must_contain", []) or []
            if all(str(term) in result["final_text"] for term in required_terms):
                output_hits += 1

            if "proactive_triggered" in expectations:
                proactive_cases += 1
                if bool(result["metadata"].get("proactive_triggered", False)) == bool(expectations["proactive_triggered"]):
                    proactive_hits += 1
            if "replan_triggered" in expectations:
                replan_cases += 1
                if bool(result["metadata"].get("replan_triggered", False)) == bool(expectations["replan_triggered"]):
                    replan_hits += 1
            if "pacing_level" in expectations:
                pacing_cases += 1
                if str(result["metadata"].get("pacing_level", "")) == str(expectations["pacing_level"]):
                    pacing_hits += 1

        total_cases = len(results)
        passed_cases = sum(1 for result in results if result["passed"])
        report = {
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "failed_cases": total_cases - passed_cases,
            "tool_recall_rate": hit_must_include / total_must_include if total_must_include else 1.0,
            "tool_precision_rate": 1.0 - (must_not_violations / total_cases if total_cases else 0.0),
            "path_reasonableness_rate": reasonable_paths / total_cases if total_cases else 1.0,
            "output_coverage_rate": output_hits / total_cases if total_cases else 1.0,
            "proactive_cases": proactive_cases,
            "proactive_hit_rate": proactive_hits / proactive_cases if proactive_cases else 1.0,
            "replan_cases": replan_cases,
            "replan_hit_rate": replan_hits / replan_cases if replan_cases else 1.0,
            "pacing_cases": pacing_cases,
            "pacing_hit_rate": pacing_hits / pacing_cases if pacing_cases else 1.0,
            "results": results,
            "failed_results": [result for result in results if not result["passed"]],
        }
        return report


async def _main() -> int:
    root = _project_root()
    runner = EvalRunner()
    report = await runner.run_all(str(root / "tests" / "eval" / "cases"))
    print(render_report(report))
    return 0 if report["passed_cases"] / max(report["total_cases"], 1) >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
