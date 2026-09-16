"""Windows-safe variant of openevolve's built-in Claude Code CLI LLM backend.

``openevolve.llm.claude_code.ClaudeCodeLLM`` (the installed package's
``provider: claude_code`` backend) appends the full user prompt as a single
trailing argv item to the ``claude -p ...`` subprocess call. That prompt
embeds the current parent program plus up to ``num_top_programs +
num_diverse_programs`` full previous-program bodies (see config.yaml) -- once
the program database has a few entries, the resulting command line reliably
exceeds Windows' ~32K character CreateProcess limit and every iteration fails
with ``OSError: [WinError 206] The filename or extension is too long``
(reproduced while setting this up: iterations 1-2 of a run succeeded while
the database was still small, then every iteration after failed once enough
programs existed to fill num_top_programs/num_diverse_programs).

This subclass is identical to the original except the prompt is piped to the
CLI's stdin instead of passed as an argv item (confirmed working: ``echo
"..." | claude -p ...`` behaves the same as the positional-arg form). Only
``--system-prompt`` stays on the command line, since it's small and fixed
size, not proportional to the evolving program's size.

Delete this file and switch ``config.yaml`` back to ``provider: claude_code``
if/when upstream openevolve fixes this (github.com/codelion/openevolve, or
wherever the installed version's source lives) -- this is a workaround for a
platform bug in the vendored backend, not something specific to this project.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess

from openevolve.llm.claude_code import ClaudeCodeLLM

logger = logging.getLogger(__name__)


class ClaudeCodeStdinLLM(ClaudeCodeLLM):
    async def generate_with_context(self, system_message: str, messages: list[dict], **kwargs) -> str:
        sys_msg = kwargs.pop("system_message", system_message) or system_message
        user_content = "\n\n".join(m.get("content", "") for m in messages if m.get("role") == "user")

        cmd = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--no-session-persistence",
            "--output-format",
            "text",
        ]
        if sys_msg:
            cmd.extend(["--system-prompt", sys_msg])

        budget = kwargs.get("max_budget_usd", self.max_budget_usd)
        cmd.extend(["--max-budget-usd", str(budget)])
        # No cmd.append(user_content) here -- that's the whole fix; it goes to
        # stdin in _run_cli_stdin below instead.

        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        loop = asyncio.get_event_loop()
        for attempt in range(retries + 1):
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: self._run_cli_stdin(cmd, user_content, timeout)),
                    timeout=timeout + 30,
                )
                return result
            except TimeoutError:
                if attempt < retries:
                    logger.warning(f"Claude Code CLI timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"Claude Code CLI error on attempt {attempt + 1}/{retries + 1}: {e}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {e}")
                    raise

    def _run_cli_stdin(self, cmd: list, stdin_text: str, timeout: int) -> str:
        try:
            result = subprocess.run(
                cmd,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.cwd,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                # The upstream backend (openevolve.llm.claude_code.ClaudeCodeLLM)
                # treats any non-empty stdout as a valid response even on a
                # non-zero exit code -- so a transient CLI failure (auth blip,
                # rate limit) gets silently accepted as "the LLM's answer",
                # which has no SEARCH/REPLACE markers and just shows up as
                # "No valid diffs found in response" with zero indication of
                # the real cause, and -- worse -- never reaches the retry loop
                # in generate_with_context, which only retries on exceptions.
                # Raising here instead makes a transient failure actually get
                # retried (retries/retry_delay from config) and, if it still
                # fails after that, surfaces the real CLI error message.
                raise RuntimeError(
                    f"Claude CLI exited {result.returncode}. "
                    f"stdout: {output[:500]!r} stderr: {result.stderr.strip()[:500]!r}"
                )
            if not output:
                raise RuntimeError(f"Empty response from Claude CLI. stderr: {result.stderr[:500]!r}")
            return output
        except subprocess.TimeoutExpired as e:
            raise TimeoutError("Claude CLI subprocess timed out") from e


def init_claude_code_stdin_client(model_cfg):
    """Factory for OpenEvolve's ``LLMModelConfig.init_client`` hook (see run.py)."""
    return ClaudeCodeStdinLLM(model_cfg)
