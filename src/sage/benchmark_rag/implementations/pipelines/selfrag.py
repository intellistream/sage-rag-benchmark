"""
Self-RAG Pipeline Implementation

Based on the Self-RAG paper (https://arxiv.org/abs/2310.11511)
Uses pre-retrieved documents from the dataset to generate answers.

This implementation uses the Self-RAG dataset format where each item contains:
- question: The question to answer
- answers: Ground truth answers
- ctxs: Pre-retrieved documents with title and text
"""

import os
from typing import Any

from openai import OpenAI

from sage.common.core import MapFunction
from sage.kernel.api.local_environment import LocalEnvironment
from sage.libs.foundation.io.sink import FileSink
from sage.libs.foundation.io.source import FileSource


class SelfRAGRetriever(MapFunction):
    """
    Self-RAG Retriever - extracts pre-retrieved documents from dataset.

    Unlike traditional retrievers that perform retrieval, this simply
    extracts the pre-computed retrieved documents from the Self-RAG dataset.
    """

    def __init__(self, config: dict):
        self.top_k = config.get("top_k", 5)

    def execute(self, item: dict[str, Any]) -> dict[str, Any]:
        """Extract pre-retrieved documents from the data item."""
        question = item["question"]
        ctxs = item.get("ctxs", [])

        # Extract top-k documents
        retrieved_docs = []
        for i, ctx in enumerate(ctxs[: self.top_k]):
            if "text" in ctx and ctx["text"].strip():
                doc = {
                    "rank": i + 1,
                    "title": ctx.get("title", ""),
                    "text": ctx["text"],
                    "score": ctx.get("score", 1.0),
                }
                retrieved_docs.append(doc)

        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "ground_truth": item.get("answers", []),
            "id": item.get("id", ""),
        }


class SelfRAGPromptor(MapFunction):
    """
    Self-RAG Promptor - builds prompts with retrieved evidence.

    Constructs prompts in the Self-RAG format with numbered evidence paragraphs.
    """

    def __init__(self, config: dict):
        self.model_name = config.get("model_name", "mistral")
        self.use_context = config.get("use_context", True)

    def execute(self, item: dict[str, Any]) -> dict[str, Any]:
        """Build prompt with evidence paragraphs."""
        question = item["question"]
        retrieved_docs = item.get("retrieved_docs", [])

        # Build evidence context
        context = None
        if self.use_context and retrieved_docs:
            evidences = []
            for doc in retrieved_docs:
                rank = doc["rank"]
                title = doc["title"]
                text = doc["text"]
                evidence = f"[{rank}] {title}\n{text}"
                evidences.append(evidence)
            context = "\n".join(evidences)

        # Build prompt based on model type
        if self.use_context and context:
            if "llama" in self.model_name.lower():
                prompt = f"[INST]{context}\n{question}[/INST]"
            else:
                prompt = f"<s>[INST]{context}\n{question}[/INST]"
        else:
            if "llama" in self.model_name.lower():
                prompt = f"[INST]{question}[/INST]"
            else:
                prompt = f"### Instruction:\n{question}\n\n### Response:\n"

        item["prompt"] = prompt
        item["context"] = context
        return item


class SelfRAGGenerator(MapFunction):
    """
    Self-RAG Generator - generates answers via OpenAI-compatible endpoint.

    默认连接本地 sageLLM 网关，也可通过 base_url/api_key 覆盖。
    """

    def __init__(self, config: dict):
        self.model_name = config.get("model_name", "mistralai/Mistral-7B-Instruct-v0.1")
        self.temperature = config.get("temperature", 0)
        self.max_tokens = config.get("max_tokens", 100)

        base_url = config.get("base_url") or os.getenv("SAGELLM_BASE_URL", "http://localhost:8901/v1")
        api_key = config.get("api_key") or os.getenv("SAGELLM_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def execute(self, item: dict[str, Any]) -> dict[str, Any]:
        """Generate answer for the question."""
        prompt = item["prompt"]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        response = completion.choices[0].message.content or ""

        # Post-process output
        response = self._postprocess(response)

        item["model_output"] = response
        return item

    def _postprocess(self, text: str) -> str:
        """Clean up model output."""
        # Take first paragraph
        text = text.split("\n\n")[0]
        # Remove end tokens
        text = text.replace("</s>", "")
        # Remove leading space
        if text and text[0] == " ":
            text = text[1:]
        return text


def process_item(item: dict[str, Any], config: dict) -> dict[str, Any]:
    """
    Process a single item through the Self-RAG pipeline.

    This is a simplified interface for benchmark runner integration.

    Args:
        item: Data item with keys: question, answers, ctxs
        config: Pipeline configuration

    Returns:
        Result dictionary with: id, question, prediction, ground_truth
    """
    # Initialize components
    retriever = SelfRAGRetriever(config)
    promptor = SelfRAGPromptor(config)
    generator = SelfRAGGenerator(config)

    # Process through pipeline
    retrieved = retriever.execute(item)
    prompted = promptor.execute(retrieved)
    result = generator.execute(prompted)

    # Format output
    return {
        "id": item.get("id", "unknown"),
        "question": item["question"],
        "prediction": result["prediction"],
        "ground_truth": item.get("answers", []),
        "retrieved_docs": result.get("retrieved_docs", []),
    }


def run_selfrag_pipeline(config_path: str):
    """
    Run Self-RAG pipeline.

    Args:
        config_path: Path to configuration YAML file
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = LocalEnvironment("selfrag_pipeline")

    (
        env.from_source(FileSource, {"file_path": config["data_path"]})
        .map(SelfRAGRetriever, config["retriever"])
        .map(SelfRAGPromptor, config["promptor"])
        .map(SelfRAGGenerator, config["generator"])
        .sink(FileSink, {"output_path": config["output_path"]})
    )

    env.submit()
    env.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Default config path
    config_dir = Path(__file__).parent.parent.parent / "config"
    config_path_default = config_dir / "config_selfrag.yaml"

    if len(sys.argv) > 1:
        config_path_arg = sys.argv[1]
    else:
        config_path_arg = str(config_path_default)

    print("🚀 Running Self-RAG Pipeline")
    print(f"📝 Config: {config_path_arg}")

    run_selfrag_pipeline(config_path_arg)

    print("✅ Self-RAG pipeline completed!")
