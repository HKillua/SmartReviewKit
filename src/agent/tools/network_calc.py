"""Deterministic network calculation tool for computer-network coursework."""

from __future__ import annotations

import ipaddress
import math
from typing import Any

from pydantic import BaseModel, Field

from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult


class NetworkCalcArgs(BaseModel):
    type: str = Field(..., description="计算类型: subnet_division/delay_throughput/crc/sliding_window/shannon_limit/nyquist_limit")
    params: dict[str, Any] = Field(default_factory=dict, description="结构化参数")


class NetworkCalcTool(Tool[NetworkCalcArgs]):
    @property
    def name(self) -> str:
        return "network_calc"

    @property
    def description(self) -> str:
        return "用于子网划分、CRC、滑动窗口、吞吐/时延、香农/奈奎斯特等精确网络计算的工具。"

    def get_args_schema(self) -> type[NetworkCalcArgs]:
        return NetworkCalcArgs

    async def execute(self, context: ToolContext, args: NetworkCalcArgs) -> ToolResult:
        calc_type = str(args.type or "").strip().lower()
        handlers = {
            "subnet_division": self._subnet_division,
            "subnet": self._subnet_division,
            "delay_throughput": self._delay_throughput,
            "crc": self._crc,
            "sliding_window": self._sliding_window,
            "shannon_limit": self._shannon_limit,
            "shannon": self._shannon_limit,
            "nyquist_limit": self._nyquist_limit,
            "nyquist": self._nyquist_limit,
        }
        handler = handlers.get(calc_type)
        if handler is None:
            return ToolResult(
                success=False,
                error=f"不支持的计算类型: {args.type}",
                metadata={"supported_types": sorted(handlers)},
            )
        try:
            payload = handler(args.params or {})
        except Exception as exc:
            return ToolResult(success=False, error=f"network_calc 计算失败: {exc}")

        return ToolResult(
            success=True,
            result_for_llm=payload["text"],
            metadata={
                "tool_output_kind": "final_answer",
                "final_response_preferred": True,
                "grounding_passthrough": True,
                "structured_result": payload["structured"],
                "calculation_type": calc_type,
            },
        )

    @staticmethod
    def _subnet_division(params: dict[str, Any]) -> dict[str, Any]:
        network_str = str(params.get("network") or "").strip()
        if not network_str:
            ip = str(params.get("ip") or "").strip()
            mask = params.get("mask")
            if not ip or mask in (None, ""):
                raise ValueError("需要提供 network 或 ip + mask")
            network_str = f"{ip}/{int(mask)}"
        network = ipaddress.ip_network(network_str, strict=False)
        num_subnets = int(params.get("num_subnets") or 0)
        if num_subnets <= 0:
            raise ValueError("num_subnets 必须大于 0")
        borrow_bits = math.ceil(math.log2(num_subnets))
        new_prefix = network.prefixlen + borrow_bits
        if new_prefix > network.max_prefixlen - 2:
            raise ValueError("子网划分后主机位不足")
        subnets = list(network.subnets(new_prefix=new_prefix))[:num_subnets]
        host_count = max(subnets[0].num_addresses - 2, 0)
        lines = [
            f"原网络: {network.with_prefixlen}",
            f"需要划分子网数: {num_subnets}",
            f"借位数: {borrow_bits} 位，新前缀长度: /{new_prefix}",
            f"每个子网可用主机数: {host_count}",
            "",
            "子网划分结果:",
        ]
        structured_subnets = []
        for index, subnet in enumerate(subnets, start=1):
            hosts = list(subnet.hosts())
            first_host = str(hosts[0]) if hosts else "-"
            last_host = str(hosts[-1]) if hosts else "-"
            lines.append(
                f"{index}. {subnet.with_prefixlen}，广播地址 {subnet.broadcast_address}，可用主机范围 {first_host} ~ {last_host}"
            )
            structured_subnets.append(
                {
                    "index": index,
                    "network": subnet.with_prefixlen,
                    "broadcast": str(subnet.broadcast_address),
                    "host_count": host_count,
                    "first_host": first_host,
                    "last_host": last_host,
                }
            )
        return {
            "text": "\n".join(lines),
            "structured": {
                "original_network": network.with_prefixlen,
                "num_subnets": num_subnets,
                "borrow_bits": borrow_bits,
                "new_prefix": new_prefix,
                "host_count_per_subnet": host_count,
                "subnets": structured_subnets,
            },
        }

    @staticmethod
    def _delay_throughput(params: dict[str, Any]) -> dict[str, Any]:
        distance_m = float(params.get("distance_m") or (float(params.get("distance_km") or 0) * 1000.0))
        propagation_speed = float(params.get("propagation_speed_mps") or 2.0e8)
        packet_bits = float(params.get("packet_bits") or params.get("frame_bits") or 0)
        rate_bps = float(params.get("rate_bps") or 0)
        if distance_m <= 0 or packet_bits <= 0 or rate_bps <= 0:
            raise ValueError("distance/packet_bits/rate_bps 必须为正数")
        propagation_delay = distance_m / propagation_speed
        transmission_delay = packet_bits / rate_bps
        rtt = 2 * propagation_delay
        window_size = float(params.get("window_size_packets") or 1.0)
        utilization = min(1.0, (window_size * transmission_delay) / max(rtt + transmission_delay, 1e-9))
        text = "\n".join(
            [
                f"传播时延: {propagation_delay:.6f} s",
                f"发送时延: {transmission_delay:.6f} s",
                f"往返时延 RTT: {rtt:.6f} s",
                f"窗口利用率（估算）: {utilization:.2%}",
            ]
        )
        return {
            "text": text,
            "structured": {
                "propagation_delay_s": propagation_delay,
                "transmission_delay_s": transmission_delay,
                "rtt_s": rtt,
                "utilization": utilization,
            },
        }

    @staticmethod
    def _crc(params: dict[str, Any]) -> dict[str, Any]:
        data = str(params.get("data_bits") or "").strip()
        generator = str(params.get("generator_bits") or "").strip()
        if not data or not generator or any(bit not in "01" for bit in data + generator):
            raise ValueError("data_bits 和 generator_bits 必须是二进制字符串")
        padded = list(data + "0" * (len(generator) - 1))
        gen = list(generator)
        working = padded[:]
        steps: list[str] = []
        for i in range(len(data)):
            if working[i] == "1":
                steps.append(f"第 {i + 1} 步：从位置 {i} 开始与生成多项式异或")
                for j in range(len(gen)):
                    working[i + j] = "0" if working[i + j] == gen[j] else "1"
        remainder = "".join(working[-(len(generator) - 1):])
        codeword = data + remainder
        text = "\n".join(
            [
                f"数据比特串: {data}",
                f"生成多项式: {generator}",
                *steps[:8],
                f"CRC 校验余数: {remainder}",
                f"发送码字: {codeword}",
            ]
        )
        return {
            "text": text,
            "structured": {
                "remainder": remainder,
                "codeword": codeword,
            },
        }

    @staticmethod
    def _sliding_window(params: dict[str, Any]) -> dict[str, Any]:
        protocol = str(params.get("protocol") or "go_back_n").strip().lower()
        seq_bits = int(params.get("seq_bits") or 3)
        window_size = int(params.get("window_size") or 1)
        base_seq = int(params.get("base_seq") or 0)
        next_seq = int(params.get("next_seq") or base_seq)
        seq_space = 2 ** seq_bits
        max_window = seq_space - 1 if protocol == "go_back_n" else max(1, seq_space // 2)
        valid = window_size <= max_window
        active = [(base_seq + offset) % seq_space for offset in range(window_size)]
        text = "\n".join(
            [
                f"协议: {protocol}",
                f"序号空间: 0 ~ {seq_space - 1}",
                f"当前窗口大小: {window_size}",
                f"该协议允许的最大窗口: {max_window}",
                f"窗口是否合法: {'是' if valid else '否'}",
                f"当前活动窗口序号: {active}",
                f"base_seq={base_seq}, next_seq={next_seq}",
            ]
        )
        return {
            "text": text,
            "structured": {
                "protocol": protocol,
                "sequence_space": seq_space,
                "max_window": max_window,
                "valid": valid,
                "active_window": active,
            },
        }

    @staticmethod
    def _shannon_limit(params: dict[str, Any]) -> dict[str, Any]:
        bandwidth_hz = float(params.get("bandwidth_hz") or 0)
        snr_db = float(params.get("snr_db") or 0)
        if bandwidth_hz <= 0:
            raise ValueError("bandwidth_hz 必须大于 0")
        snr_linear = 10 ** (snr_db / 10.0)
        capacity = bandwidth_hz * math.log2(1 + snr_linear)
        text = (
            f"带宽: {bandwidth_hz:.2f} Hz\n"
            f"SNR: {snr_db:.2f} dB（线性值 {snr_linear:.4f}）\n"
            f"香农极限信道容量: {capacity:.2f} bps"
        )
        return {
            "text": text,
            "structured": {"capacity_bps": capacity, "snr_linear": snr_linear},
        }

    @staticmethod
    def _nyquist_limit(params: dict[str, Any]) -> dict[str, Any]:
        bandwidth_hz = float(params.get("bandwidth_hz") or 0)
        levels = float(params.get("signal_levels") or params.get("levels") or 0)
        if bandwidth_hz <= 0 or levels <= 1:
            raise ValueError("bandwidth_hz 必须大于 0 且 signal_levels 必须大于 1")
        capacity = 2 * bandwidth_hz * math.log2(levels)
        text = (
            f"带宽: {bandwidth_hz:.2f} Hz\n"
            f"信号电平数: {levels:.0f}\n"
            f"奈奎斯特极限速率: {capacity:.2f} bps"
        )
        return {
            "text": text,
            "structured": {"capacity_bps": capacity},
        }
