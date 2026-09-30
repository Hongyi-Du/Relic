"""Team — multi-agent orchestration (sequential / parallel)."""
from __future__ import annotations

import concurrent.futures
from typing import Callable, List, Optional

from .agent import Agent


class Team:
    """
    Orchestrate multiple Agents on a shared task.

    Strategies:
        - "sequential": agents run in order, each gets previous output
        - "parallel": agents run concurrently, results summarized
    """

    def __init__(
        self,
        agents: List[Agent],
        strategy: str = "sequential",
    ):
        if not agents:
            raise ValueError("Team requires at least one agent")
        self.agents = agents
        self.strategy = strategy

    def run(self, task: str) -> str:
        if self.strategy == "sequential":
            return self._run_sequential(task)
        elif self.strategy == "parallel":
            return self._run_parallel(task)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _run_sequential(self, task: str) -> str:
        """Chain agents: each gets the previous agent's output."""
        current = task
        for i, agent in enumerate(self.agents):
            if i == 0:
                current = agent.chat(current)
            else:
                prompt = (
                    f"Previous agent ({self.agents[i-1].name}) produced:\n\n"
                    f"{current}\n\n"
                    f"Original task: {task}\n\n"
                    f"Continue based on the above."
                )
                current = agent.chat(prompt)
        return current

    def _run_parallel(self, task: str) -> str:
        """Run all agents concurrently, then summarize results."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(agent.chat, task): agent
                for agent in self.agents
            }
            results = []
            for future in concurrent.futures.as_completed(futures):
                agent = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(f"[{agent.name} failed: {e}]")

        return self._summarize(results, task)

    def _summarize(self, results: List[str], task: str) -> str:
        """Summarize parallel results. Uses first agent's LLM if available."""
        if len(results) == 1:
            return results[0]

        # Build summary prompt
        parts = []
        for i, (agent, result) in enumerate(zip(self.agents, results)):
            parts.append(f"## {agent.name}'s response:\n{result}")
        all_responses = "\n\n".join(parts)

        summary_prompt = (
            f"Multiple agents worked on this task: {task}\n\n"
            f"{all_responses}\n\n"
            f"Synthesize these responses into a single coherent result."
        )

        # Use first agent's LLM to summarize
        llm = getattr(self.agents[0], "_llm", None)
        if llm:
            response = llm.create_completion_with_retry(
                messages=[{"role": "user", "content": summary_prompt}]
            )
            if response and response.get("content"):
                return response["content"]

        # Fallback: concatenate
        return "\n\n---\n\n".join(results)
