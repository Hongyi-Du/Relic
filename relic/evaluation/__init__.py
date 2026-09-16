"""Public evaluation primitives for Relic experiments."""

from .execution import ExecutionPolicy, build_command_executor
from .functional_overlap import functional_overlap
from .time_machine import (
    build_time_machine_evaluation_plan,
    evaluate_time_machine_candidate,
    oracle_blocking_reasons,
)

__all__ = [
    "ExecutionPolicy",
    "build_command_executor",
    "build_time_machine_evaluation_plan",
    "evaluate_time_machine_candidate",
    "functional_overlap",
    "oracle_blocking_reasons",
]
