"""Swappable LLM backends.

  SYCO_LLM_BACKEND=cli  -> Claude Code CLI in print mode, using your claude.ai login. PERSONAL PROTOTYPING ONLY:
                           a consumer subscription must not power a service used by other people.
  SYCO_LLM_BACKEND=api  -> Anthropic API (ANTHROPIC_API_KEY). Use this for anything others will use.

Both expose: complete(system, messages) -> LLMResult
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from . import config


@dataclass
class LLMResult:
    text: str
    stop_reason: str = "end_turn"
    model: str = ""
    error: str = ""


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict], max_tokens: int = 16000) -> LLMResult: ...
    def stream(self, system: str, messages: list[dict], max_tokens: int = 16000) -> Iterator[str]: ...


# ---------------------------------------------------------------------------
def _find_claude_cli() -> str:
    if config.CLAUDE_CLI:
        return config.CLAUDE_CLI
    found = shutil.which("claude")
    if found:
        return found
    # Claude Code VS Code extension ships the CLI binary
    pattern = str(Path.home() / ".vscode" / "extensions" / "anthropic.claude-code-*" / "resources" / "native-binary" / "claude*")
    candidates = [c for c in glob.glob(pattern) if c.lower().endswith(("claude", "claude.exe"))]
    if not candidates:
        raise RuntimeError("Claude CLI not found. Install Claude Code or set SYCO_CLAUDE_CLI to the claude executable.")

    def version_key(p: str) -> tuple:
        ver = Path(p).parents[2].name.split("-")[2] if "-" in Path(p).parents[2].name else "0"
        return tuple(int(x) for x in ver.split(".") if x.isdigit())

    return max(candidates, key=version_key)


def _transcript(messages: list[dict]) -> str:
    """CLI print mode is single-turn: render the conversation history into one prompt."""
    if len(messages) == 1:
        return messages[0]["content"]
    parts = ["<conversation_history>"]
    for m in messages[:-1]:
        role = "Psychologist" if m["role"] == "user" else "Assistant"
        parts.append(f"<{role}>\n{m['content']}\n</{role}>")
    parts.append("</conversation_history>\n")
    parts.append(messages[-1]["content"])
    return "\n".join(parts)


class ClaudeCLI:
    def __init__(self, model: str = config.LLM_MODEL):
        self.exe = _find_claude_cli()
        self.model = model
        # run in an empty dir so no project CLAUDE.md / settings leak into the prompt
        self.workdir = Path(tempfile.gettempdir()) / "syco_cli_cwd"
        self.workdir.mkdir(exist_ok=True)

    def complete(self, system: str, messages: list[dict], max_tokens: int = 16000) -> LLMResult:
        cmd = [
            self.exe, "-p",
            "--system-prompt", system,
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--output-format", "json",
            "--model", self.model,
        ]
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}  # allow running from inside a Claude Code session
        try:
            proc = subprocess.run(
                cmd, input=_transcript(messages), capture_output=True, text=True, encoding="utf-8",
                cwd=self.workdir, env=env, timeout=config.LLM_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return LLMResult("", "error", self.model, "Claude CLI timed out")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return LLMResult("", "error", self.model, (proc.stderr or proc.stdout or "no output")[:2000])
        if data.get("is_error"):
            return LLMResult(data.get("result", ""), "error", self.model, data.get("result") or "CLI error")
        return LLMResult(data.get("result", ""), data.get("stop_reason") or "end_turn", self.model)

    def stream(self, system: str, messages: list[dict], max_tokens: int = 16000) -> Iterator[str]:
        res = self.complete(system, messages, max_tokens)
        if res.error:
            raise RuntimeError(res.error)
        yield res.text


# ---------------------------------------------------------------------------
class AnthropicAPI:
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model: str = config.LLM_MODEL):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model

    def _params(self, system: str, messages: list[dict], max_tokens: int) -> dict:
        return dict(
            model=self.model,
            max_tokens=max_tokens,
            # stable system prompt cached; retrieved context lives in the user turn
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config={"effort": config.LLM_EFFORT},
            betas=[self.FALLBACK_BETA],
            fallbacks="default",  # on a safety refusal, retry server-side on a fallback model
        )

    def complete(self, system: str, messages: list[dict], max_tokens: int = 16000) -> LLMResult:
        with self.client.beta.messages.stream(**self._params(system, messages, max_tokens)) as s:
            msg = s.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text")
        if msg.stop_reason == "refusal":
            return LLMResult(text, "refusal", msg.model, "The model declined to answer this request.")
        return LLMResult(text, msg.stop_reason or "end_turn", msg.model)

    def stream(self, system: str, messages: list[dict], max_tokens: int = 16000) -> Iterator[str]:
        with self.client.beta.messages.stream(**self._params(system, messages, max_tokens)) as s:
            yield from s.text_stream
            msg = s.get_final_message()
        if msg.stop_reason == "refusal":
            yield "\n\n_(The model declined to continue this answer.)_"


def get_llm(backend: str | None = None) -> LLM:
    backend = (backend or config.LLM_BACKEND).lower()
    if backend == "api":
        return AnthropicAPI()
    if backend == "cli":
        return ClaudeCLI()
    raise ValueError(f"Unknown SYCO_LLM_BACKEND={backend!r} (use 'cli' or 'api')")
