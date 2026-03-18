"""Deterministic protocol state simulator for computer-network coursework."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult


class ProtocolStateSimulatorArgs(BaseModel):
    protocol: str = Field(..., description="模拟协议类型: tcp_handshake/tcp_teardown/tcp_congestion/rip_update")
    params: dict[str, Any] = Field(default_factory=dict, description="结构化模拟参数")
    fault_injection: dict[str, Any] = Field(default_factory=dict, description="故障注入配置，如丢包、超时、乱序")


class ProtocolStateSimulatorTool(Tool[ProtocolStateSimulatorArgs]):
    @property
    def name(self) -> str:
        return "protocol_state_simulator"

    @property
    def description(self) -> str:
        return "模拟 TCP 握手/挥手、拥塞控制和路由协议等过程，输出结构化状态变化序列。"

    def get_args_schema(self) -> type[ProtocolStateSimulatorArgs]:
        return ProtocolStateSimulatorArgs

    async def execute(
        self,
        context: ToolContext,
        args: ProtocolStateSimulatorArgs,
    ) -> ToolResult:
        protocol = str(args.protocol or "").strip().lower()
        handlers = {
            "tcp_handshake": self._tcp_handshake,
            "tcp_teardown": self._tcp_teardown,
            "tcp_congestion": self._tcp_congestion,
            "rip_update": self._rip_update,
        }
        handler = handlers.get(protocol)
        if handler is None:
            return ToolResult(
                success=False,
                error=f"不支持的协议模拟类型: {args.protocol}",
                metadata={"supported_protocols": sorted(handlers)},
            )
        try:
            payload = handler(args.params or {}, args.fault_injection or {})
        except Exception as exc:
            return ToolResult(success=False, error=f"protocol_state_simulator 模拟失败: {exc}")
        return ToolResult(
            success=True,
            result_for_llm=payload["text"],
            metadata={
                "tool_output_kind": "analysis_context",
                "simulation_protocol": protocol,
                "simulation_steps": payload["steps"],
                "mermaid": payload.get("mermaid", ""),
            },
        )

    @staticmethod
    def _tcp_handshake(params: dict[str, Any], fault: dict[str, Any]) -> dict[str, Any]:
        seq = int(params.get("initial_seq") or 100)
        server_seq = int(params.get("server_seq") or 400)
        drop_packet = int(fault.get("drop_packet") or 0)
        timeout = bool(fault.get("timeout") or drop_packet == 2)

        steps = [
            {
                "step": 1,
                "from": "Client",
                "to": "Server",
                "segment": "SYN",
                "seq": seq,
                "ack": None,
                "client_state": "SYN-SENT",
                "server_state": "LISTEN",
                "note": "客户端发起连接请求",
            },
            {
                "step": 2,
                "from": "Server",
                "to": "Client",
                "segment": "SYN-ACK",
                "seq": server_seq,
                "ack": seq + 1,
                "client_state": "SYN-SENT",
                "server_state": "SYN-RECEIVED",
                "note": "服务器确认并同步自己的初始序号",
            },
            {
                "step": 3,
                "from": "Client",
                "to": "Server",
                "segment": "ACK",
                "seq": seq + 1,
                "ack": server_seq + 1,
                "client_state": "ESTABLISHED",
                "server_state": "ESTABLISHED",
                "note": "客户端确认，连接建立",
            },
        ]

        if drop_packet == 2:
            steps.insert(
                2,
                {
                    "step": 3,
                    "from": "Network",
                    "to": "Client",
                    "segment": "DROP",
                    "seq": server_seq,
                    "ack": seq + 1,
                    "client_state": "SYN-SENT",
                    "server_state": "SYN-RECEIVED",
                    "note": "SYN-ACK 丢失，客户端收不到确认包",
                },
            )
            steps.append(
                {
                    "step": 4,
                    "from": "Client",
                    "to": "Server",
                    "segment": "RETRANSMIT SYN",
                    "seq": seq,
                    "ack": None,
                    "client_state": "SYN-SENT",
                    "server_state": "SYN-RECEIVED",
                    "note": "客户端超时后重传 SYN",
                }
            )

        step_lines = []
        for item in steps:
            ack_suffix = ""
            if item["ack"] is not None:
                ack_suffix = f", ack={item['ack']}"
            step_lines.append(
                f"{item['step']}. {item['from']} -> {item['to']} 发送 {item['segment']}"
                f" (seq={item['seq']}{ack_suffix})；"
                f" 客户端状态={item['client_state']}，服务端状态={item['server_state']}；{item['note']}"
            )
        lines = ["TCP 三次握手模拟：", *step_lines]
        if timeout:
            lines.append("故障结论：由于 SYN-ACK 丢失，客户端会在超时后重传 SYN，直到收到确认或达到重试上限。")

        mermaid = "\n".join(
            [
                "sequenceDiagram",
                "participant Client",
                "participant Server",
                f"Client->>Server: SYN seq={seq}",
                (
                    f"Server--xClient: SYN-ACK seq={server_seq}, ack={seq + 1}"
                    if drop_packet == 2
                    else f"Server->>Client: SYN-ACK seq={server_seq}, ack={seq + 1}"
                ),
                f"Client->>Server: ACK seq={seq + 1}, ack={server_seq + 1}",
            ]
        )
        return {"text": "\n".join(lines), "steps": steps, "mermaid": mermaid}

    @staticmethod
    def _tcp_teardown(params: dict[str, Any], fault: dict[str, Any]) -> dict[str, Any]:
        client_seq = int(params.get("client_seq") or 800)
        server_seq = int(params.get("server_seq") or 1600)
        steps = [
            {
                "step": 1,
                "from": "Client",
                "to": "Server",
                "segment": "FIN",
                "seq": client_seq,
                "ack": server_seq,
                "client_state": "FIN-WAIT-1",
                "server_state": "ESTABLISHED",
                "note": "客户端请求关闭发送方向",
            },
            {
                "step": 2,
                "from": "Server",
                "to": "Client",
                "segment": "ACK",
                "seq": server_seq,
                "ack": client_seq + 1,
                "client_state": "FIN-WAIT-2",
                "server_state": "CLOSE-WAIT",
                "note": "服务器确认收到 FIN",
            },
            {
                "step": 3,
                "from": "Server",
                "to": "Client",
                "segment": "FIN",
                "seq": server_seq,
                "ack": client_seq + 1,
                "client_state": "FIN-WAIT-2",
                "server_state": "LAST-ACK",
                "note": "服务器处理完剩余数据后发送 FIN",
            },
            {
                "step": 4,
                "from": "Client",
                "to": "Server",
                "segment": "ACK",
                "seq": client_seq + 1,
                "ack": server_seq + 1,
                "client_state": "TIME-WAIT",
                "server_state": "CLOSED",
                "note": "客户端确认后进入 TIME-WAIT，等待 2MSL",
            },
        ]
        lines = ["TCP 四次挥手模拟："]
        lines.extend(
            f"{item['step']}. {item['from']} -> {item['to']} {item['segment']}，客户端={item['client_state']}，服务端={item['server_state']}；{item['note']}"
            for item in steps
        )
        if fault.get("fin_retransmit"):
            lines.append("故障结论：若最后 ACK 丢失，服务端会重传 FIN，客户端在 TIME-WAIT 中再次确认。")
        return {"text": "\n".join(lines), "steps": steps}

    @staticmethod
    def _tcp_congestion(params: dict[str, Any], fault: dict[str, Any]) -> dict[str, Any]:
        cwnd = int(params.get("initial_cwnd") or 1)
        ssthresh = int(params.get("ssthresh") or 16)
        events = list(params.get("events") or ["normal", "normal", "timeout"])
        steps = []
        phase = "slow_start" if cwnd < ssthresh else "congestion_avoidance"
        for index, event in enumerate(events, start=1):
            before = cwnd
            before_phase = phase
            if event == "normal":
                if cwnd < ssthresh:
                    cwnd *= 2
                    phase = "slow_start" if cwnd < ssthresh else "congestion_avoidance"
                else:
                    cwnd += 1
                    phase = "congestion_avoidance"
                note = "正常收到 ACK，窗口按当前阶段增长"
            elif event == "triple_dup_ack":
                ssthresh = max(cwnd // 2, 2)
                cwnd = ssthresh + 3
                phase = "fast_recovery"
                note = "收到三个重复 ACK，进入快速恢复"
            else:
                ssthresh = max(cwnd // 2, 2)
                cwnd = 1
                phase = "slow_start"
                note = "发生超时，ssthresh 减半并回到慢启动"
            steps.append(
                {
                    "step": index,
                    "event": event,
                    "before_cwnd": before,
                    "after_cwnd": cwnd,
                    "ssthresh": ssthresh,
                    "phase_before": before_phase,
                    "phase_after": phase,
                    "note": note,
                }
            )

        lines = ["TCP 拥塞控制模拟："]
        lines.extend(
            f"{item['step']}. 事件={item['event']}，cwnd: {item['before_cwnd']} -> {item['after_cwnd']}，"
            f"ssthresh={item['ssthresh']}，阶段: {item['phase_before']} -> {item['phase_after']}；{item['note']}"
            for item in steps
        )
        if fault:
            lines.append(f"故障注入: {fault}")
        return {"text": "\n".join(lines), "steps": steps}

    @staticmethod
    def _rip_update(params: dict[str, Any], fault: dict[str, Any]) -> dict[str, Any]:
        initial_distance = int(params.get("initial_distance") or 1)
        update_distance = int(params.get("update_distance") or 3)
        next_hop = str(params.get("next_hop") or "RouterB")
        poisoned = bool(fault.get("poison_reverse"))
        steps = [
            {
                "step": 1,
                "router": "RouterA",
                "route": "目的网段",
                "distance": initial_distance,
                "next_hop": next_hop,
                "note": "初始距离向量",
            },
            {
                "step": 2,
                "router": "RouterA",
                "route": "目的网段",
                "distance": update_distance,
                "next_hop": next_hop,
                "note": "收到邻居更新后调整距离",
            },
        ]
        if poisoned:
            steps.append(
                {
                    "step": 3,
                    "router": "RouterA",
                    "route": "目的网段",
                    "distance": 16,
                    "next_hop": next_hop,
                    "note": "触发毒性逆转，向原邻居通告无穷大距离",
                }
            )
        lines = ["RIP 路由更新模拟："]
        lines.extend(
            f"{item['step']}. {item['router']} 记录 {item['route']} 距离={item['distance']}，下一跳={item['next_hop']}；{item['note']}"
            for item in steps
        )
        return {"text": "\n".join(lines), "steps": steps}
