"""LLM-as-judge scoring with a Lithuanian-specific rubric."""

from __future__ import annotations

from dataclasses import dataclass

from gintaras.backends import Backend, parse_json_object
from gintaras.lt import lt_score

Message = dict[str, str]

JUDGE_TEMPLATE = """You are a strict expert evaluator of Lithuanian-language AI assistant responses. \
You are a native Lithuanian speaker and an experienced Lithuanian language teacher and editor.

Score the CANDIDATE RESPONSE to the last user turn on a 1-10 scale for each criterion:
- faithfulness: factual correctness. If the user provided a text/context, every claim must be supported by it, \
and the response must say the information is absent when it is. Invented facts are severe errors.
- language: grammar, spelling, Lithuanian letters (ą č ę ė į š ų ū ž), punctuation, agreement, natural idiomatic \
Lithuanian. Penalize anglicisms, calques, barbarisms, machine-translation style and mixed languages. \
A response that is not in Lithuanian (when Lithuanian was expected) gets language = 1.
- helpfulness: fully and directly addresses the request, follows every constraint (length, format, style), \
well structured, no padding or needless disclaimers.
- overall: holistic quality. 10 = flawless, publishable as is; 8 = good with minor issues; \
5 = noticeable errors; 1 = useless.
{reference}
### CONVERSATION
{conversation}

### CANDIDATE RESPONSE
{response}

Return ONLY a JSON object: {{"faithfulness": n, "language": n, "helpfulness": n, "overall": n, \
"issues": "<one short sentence in English>"}}"""

CRITERIA = ("faithfulness", "language", "helpfulness", "overall")


@dataclass
class Score:
    faithfulness: float
    language: float
    helpfulness: float
    overall: float
    issues: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def render_conversation(messages: list[Message]) -> str:
    parts = []
    for m in messages:
        if m["role"] == "system":
            continue
        who = "USER" if m["role"] == "user" else "ASSISTANT"
        parts.append(f"[{who}]\n{m['content']}")
    return "\n\n".join(parts)


def build_judge_prompt(messages: list[Message], response: str, reference: str | None = None) -> str:
    ref = ""
    if reference:
        ref = f"\nA reference answer is provided; the candidate need not match it word for word.\n### REFERENCE\n{reference}\n"
    return JUDGE_TEMPLATE.format(
        reference=ref, conversation=render_conversation(messages), response=response
    )


def parse_score(text: str | None) -> Score | None:
    obj = parse_json_object(text)
    if not obj:
        return None
    try:
        vals = {k: max(1.0, min(10.0, float(obj[k]))) for k in CRITERIA}
    except (KeyError, TypeError, ValueError):
        return None
    return Score(**vals, issues=str(obj.get("issues", ""))[:500])


def judge_many(
    judge: Backend,
    items: list[tuple[list[Message], str, str | None]],
    expect_lithuanian: bool = True,
) -> list[Score | None]:
    """Score (conversation, response, reference) triples. Responses that are
    clearly not Lithuanian are capped regardless of what the judge says."""
    prompts = [[{"role": "user", "content": build_judge_prompt(m, r, ref)}] for m, r, ref in items]
    raw = judge.chat_many(prompts, temperature=0.0, max_tokens=512)
    scores = [parse_score(t) for t in raw]
    if expect_lithuanian:
        for i, (s, (_, resp, _)) in enumerate(zip(scores, items)):
            if s is not None and len(resp) > 80 and lt_score(resp) < 0.4:
                scores[i] = Score(s.faithfulness, 1.0, s.helpfulness, min(s.overall, 3.0), "not Lithuanian (auto)")
    return scores
