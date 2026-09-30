"""Thin wrapper so the LLM runner can call an Anthropic model.

Needs `pip install anthropic` and ANTHROPIC_API_KEY set in the environment.
To evaluate some other LLM, write a function with the same signature:
    generate(question, context) -> {"text", "input_tokens", "output_tokens"}
"""
import anthropic

SYSTEM = (
    "Answer the question briefly. If a context is given, use only the context. "
    "If the context does not contain the answer, say you cannot tell from the context."
)


def make_generate(model, max_tokens=200, temperature=None):
    client = anthropic.Anthropic()

    def generate(question, context=""):
        prompt = f"Context:\n{context}\n\nQuestion: {question}" if context else question
        kwargs = {"temperature": temperature} if temperature is not None else {}
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return {
            "text": text,
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }

    return generate
