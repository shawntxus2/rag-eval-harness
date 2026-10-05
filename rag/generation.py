"""Generation provider seam (Phase 5).

One entry point — :func:`generate` — takes a prompt and the retrieved context
chunks and returns the model's answer plus token counts. The provider is chosen
by the ``GENERATION_PROVIDER`` config value; only ``openai`` is implemented
today. Adding ``anthropic`` later means filling in one branch here and changing
nothing at the call site.

Deliberately thin: a dataclass result and a dispatch function, no client
classes or registries. Do not grow this past "pick a provider, make one call".

NOTE: retrieval evaluation (Phases 1-4) does not call this. Generation is a
separate, later phase; recall/latency numbers never depend on an API key.
"""
from __future__ import annotations

from dataclasses import dataclass

from rag.config import settings

_SYSTEM = (
    "You answer strictly from the numbered context passages provided. "
    "Cite the passage number(s) you used in square brackets, e.g. [2]. "
    "If the passages do not contain the answer, say so plainly."
)


@dataclass(frozen=True)
class Generation:
    """Result of one generation call."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str  # the model id the provider reported actually using


def _render(prompt: str, context_chunks: list[str]) -> tuple[str, str]:
    """Return (system, user) message text for the given prompt + context."""
    passages = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context_chunks))
    user = f"Context passages:\n\n{passages}\n\nQuestion: {prompt}"
    return _SYSTEM, user


def generate(
    prompt: str,
    context_chunks: list[str],
    *,
    max_output_tokens: int = 1024,
) -> Generation:
    """Answer ``prompt`` using ``context_chunks``, via the configured provider.

    Fails loudly: a missing/rejected key or an unknown provider raises. There is
    no stub fallback — a silent empty answer would corrupt any evaluation.
    """
    provider = settings.generation_provider
    if provider == "openai":
        return _openai(prompt, context_chunks, max_output_tokens)
    if provider == "anthropic":
        raise NotImplementedError(
            "GENERATION_PROVIDER='anthropic' is a reserved seam, not yet wired. "
            "Implement an _anthropic() branch here; the call site need not change."
        )
    raise ValueError(
        f"Unknown GENERATION_PROVIDER {provider!r}. Supported: 'openai' "
        "(and 'anthropic' once implemented)."
    )


def _openai(prompt: str, context_chunks: list[str], max_output_tokens: int) -> Generation:
    # Imported lazily so DB/retrieval code paths don't require the SDK installed.
    from openai import OpenAI

    system, user = _render(prompt, context_chunks)
    client = OpenAI()  # reads OPENAI_API_KEY from the environment; raises if absent
    resp = client.chat.completions.create(
        model=settings.generation_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_output_tokens,
    )
    usage = resp.usage
    return Generation(
        text=resp.choices[0].message.content or "",
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        model=resp.model,
    )
