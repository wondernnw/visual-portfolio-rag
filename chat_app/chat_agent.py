"""Chat agents for portfolio evaluation.

- GroqChatAgent: Uses Groq's native function calling to orchestrate MCP tools.
- DeterministicAgent: Button-driven workflow for local mode (no LLM orchestration).
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional

from chat_app.mcp_client import MCPBridge
from chat_app.prompts import SYSTEM_PROMPT_GROQ


class EventType(Enum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"


@dataclass
class ChatEvent:
    """Event yielded by agents during processing."""

    type: EventType
    content: str = ""
    tool_name: str = ""
    tool_args: Dict = field(default_factory=dict)
    tool_result: Any = None


class GroqChatAgent:
    """Chat agent using Groq's function calling to orchestrate MCP tools.

    Maintains conversation history and lets the LLM decide which tools
    to call based on user messages.
    """

    def __init__(self, api_key: str, bridge: MCPBridge, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model
        self._bridge = bridge
        self._tools = bridge.get_tool_schemas()
        self._history: List[Dict] = [
            {"role": "system", "content": SYSTEM_PROMPT_GROQ}
        ]

    @property
    def history(self) -> List[Dict]:
        """Get conversation history (for display)."""
        return self._history

    def set_history(self, history: List[Dict]):
        """Restore conversation history (from session state)."""
        self._history = history

    def chat(self, user_message: str) -> Generator[ChatEvent, None, None]:
        """Process a user message, yielding events as they occur.

        The LLM may call multiple tools in sequence before responding.
        """
        self._history.append({"role": "user", "content": user_message})

        max_iterations = 10
        for _ in range(max_iterations):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._history,
                tools=self._tools if self._tools else None,
                temperature=0.3,
            )

            msg = response.choices[0].message

            # If the model wants to call tools
            if msg.tool_calls:
                # Add assistant message with tool calls to history
                self._history.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # Execute each tool call
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    yield ChatEvent(
                        type=EventType.TOOL_CALL,
                        tool_name=tool_name,
                        tool_args=tool_args,
                    )

                    try:
                        result = self._bridge.call_tool(tool_name, tool_args)
                    except Exception as e:
                        result = {"status": "error", "message": str(e)}

                    yield ChatEvent(
                        type=EventType.TOOL_RESULT,
                        tool_name=tool_name,
                        tool_result=result,
                    )

                    # Add tool result to history
                    self._history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })

                # Continue loop so LLM can process tool results
                continue

            # No tool calls -- final text response
            text = msg.content or ""
            self._history.append({"role": "assistant", "content": text})
            yield ChatEvent(type=EventType.TEXT, content=text)
            break


class DeterministicAgent:
    """Button-driven agent for predictable step-by-step pipeline execution.

    Used by sidebar buttons to run tools in a fixed sequence
    without LLM orchestration.
    """

    def __init__(self, bridge: MCPBridge):
        self._bridge = bridge

    def run_step(self, tool_name: str, arguments: Optional[Dict] = None) -> Generator[ChatEvent, None, None]:
        """Execute a specific tool and yield events."""
        arguments = arguments or {}

        yield ChatEvent(
            type=EventType.TOOL_CALL,
            tool_name=tool_name,
            tool_args=arguments,
        )

        try:
            result = self._bridge.call_tool(tool_name, arguments)
        except Exception as e:
            yield ChatEvent(
                type=EventType.ERROR,
                content=str(e),
                tool_name=tool_name,
            )
            return

        yield ChatEvent(
            type=EventType.TOOL_RESULT,
            tool_name=tool_name,
            tool_result=result,
        )

    def run_full_pipeline(
        self,
        bundle_path: str,
        checkliste_path: Optional[str] = None,
    ) -> Generator[ChatEvent, None, None]:
        """Run the full evaluation pipeline step by step.

        Loads a pre-computed bundle, optionally loads checkliste images,
        evaluates all criteria via Groq, and generates a PDF report.
        """
        # Step 1: Load retrieval bundle
        yield from self.run_step("load_bundle", {"path": bundle_path})

        # Step 2: Load checkliste images (optional, for evaluation context)
        if checkliste_path:
            yield from self.run_step("load_checkliste", {"path": checkliste_path})

        # Step 3: Evaluate all criteria
        yield from self.run_step("evaluate_all", {})

        # Step 4: Generate report
        yield from self.run_step("generate_report", {"output_path": ""})
