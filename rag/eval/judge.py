from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

JUDGE_PROMPT = ChatPromptTemplate.from_template("""
You are a STRICT evaluator of a Retrieval-Augmented Generation (RAG) system.

Your job is to FIND FAULTS.

You MUST start each score at 5 and DEDUCT points if ANY issue exists.

Scoring rules:
- 5 = Perfect, no issues at all
- 4 = Minor issue (verbosity, slight redundancy)
- 3 = Noticeable issue (missing detail, weak grounding)
- 2 = Serious issue (partial hallucination, weak answer)
- 1 = Critical failure (hallucination or wrong answer)

You MUST be conservative.
If you are unsure, DEDUCT points.

Evaluate ONLY using the provided context.
If the answer includes ANY information not in the context, deduct at least 2 points for faithfulness.

Evaluate the following:

1. Faithfulness:
- Does the answer strictly use only the provided context?
- Penalize ANY speculation or inferred knowledge.

2. Relevance:
- Does the answer directly answer the question?
- Penalize fluff or off-topic text.

3. Completeness:
- Does the answer cover all key points present in the context?
- Penalize missing steps or concepts.

4. Overall Quality:
- Based on the above scores.

Return ONLY valid JSON in this exact format:
{{
  "faithfulness": <int>,
  "relevance": <int>,
  "completeness": <int>,
  "overall_quality": <int>,
  "reasoning": "<short justification explaining any deductions>"
}}

Context:
{context}

Question:
{question}

Answer:
{answer}
""")


judge_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)


import json
import re

def judge_answer(question: str, context: str, answer: str) -> dict:
    messages = JUDGE_PROMPT.format_messages(
        context=context,
        question=question,
        answer=answer
    )

    response = judge_llm.invoke(messages)
    content = response.content.strip()

    # --------------------------------------------------
    # 1️⃣ Try direct JSON parse
    # --------------------------------------------------
    try:
        return json.loads(content)
    except Exception:
        pass

    # --------------------------------------------------
    # 2️⃣ Try extracting JSON from markdown fences
    # --------------------------------------------------
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
        pass

    # --------------------------------------------------
    # 3️⃣ Final fallback (keep raw reasoning)
    # --------------------------------------------------
    return {
        "faithfulness": None,
        "relevance": None,
        "completeness": None,
        "overall_quality": None,
        "reasoning": content
    }
