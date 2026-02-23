import os
os.environ["PYDANTIC_V1_FORCE_DISABLE"] = "1"

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import json


llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0,
    num_ctx=1024  
)


qa_prompt = PromptTemplate.from_template("""
You are a factual QA system.

Using ONLY the paragraph below, answer the question.

Paragraph:
{paragraph}

Question:
{question}

Return ONLY the exact answer. Do not explain.
""")

def generate_correct_answer(paragraph, question):
    chain = qa_prompt | llm
    response = chain.invoke({
        "paragraph": paragraph,
        "question": question
    })
    return response.content.strip()


class HallucinationOutput(BaseModel):
    hallucination_types: List[str]
    explanation: str
    corrected_answer: str


hall_parser = JsonOutputParser(pydantic_object=HallucinationOutput)

hall_prompt = PromptTemplate.from_template("""
You are an AI hallucination detector.

MULTI-LABEL CLASSIFICATION IS ALLOWED.
You may assign multiple hallucination types if applicable.

Definitions:

1A → Entity out of context (answer contains entity not present in paragraph)
1B → Entity tuple mismatch (entities exist but incorrectly paired)
2A → Intent mismatch (meaning reversed or distorted)
3A → Triple semantic mismatch (Subject-Predicate-Object incorrect)
NONE → If answer is fully correct

Analyze carefully.

Paragraph:
{paragraph}

Question:
{question}

Final Answer:
{answer}

Correct Answer:
{correct_answer}


Return ONLY valid JSON:
{format_instructions}
""")

hall_chain = (
    hall_prompt.partial(
        format_instructions=hall_parser.get_format_instructions()
    )
    | llm
    | hall_parser
)

def detect_hallucination(paragraph, question, answer):

    correct_answer = generate_correct_answer(paragraph, question)


    try:
        result = hall_chain.invoke({
            "paragraph": paragraph,
            "question": question,
            "answer": answer,
            "correct_answer": correct_answer,
        })

        if not result.get("hallucination_types"):
            result["hallucination_types"] = ["NONE"]

        return result

    except Exception as e:
        return {
            "hallucination_types": ["ERROR"],
            "explanation": f"Failed to parse model output: {str(e)}",
            "corrected_answer": correct_answer
        }


if __name__ == "__main__":

    print("\n===== Hallucination Detection System (Multi-Label) =====\n")

    while True:

        print("\nEnter Paragraph (type END on a new line to finish):")

        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)

        paragraph = "\n".join(lines)

        question = input("\nEnter Question:\n")
        answer = input("\nEnter Model Answer:\n")

        print("\nRunning Detection...\n")

        result = detect_hallucination(paragraph, question, answer)

        print("===== RESULT =====")
        print(json.dumps(result, indent=4))

        again = input("\nDo you want to test another example? (y/n): ")
        if again.lower() != "y":
            print("\nExiting System. Goodbye.\n")
            break