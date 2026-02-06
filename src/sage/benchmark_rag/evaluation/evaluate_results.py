import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sage.common.config.output_paths import get_output_file

# ============================================================================
# 文本标准化模块
# ============================================================================

# # 英文常见停顿词/停用词列表
STOP_WORDS: set[str] = set()
#     'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
#     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
#     'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
#     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
#     'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
#     'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how',
#     'what', 'which', 'who', 'whom', 'whose', 'if', 'then', 'else', 'so', 'as', 'than',
#     'not', 'no', 'yes', 'all', 'any', 'some', 'each', 'every', 'other', 'another',
#     'more', 'most', 'less', 'least', 'much', 'many', 'few', 'little', 'very', 'quite',
#     'just', 'only', 'also', 'too', 'even', 'still', 'yet', 'already', 'again',
#     'up', 'down', 'out', 'off', 'over', 'under', 'above', 'below', 'through', 'between',
#     'into', 'onto', 'from', 'within', 'without', 'during', 'before', 'after', 'since', 'until'
# }


def normalize_text_basic(text: str) -> str:
    """
    基础文本标准化（用于简单匹配）
    Args:
        text: 原始文本
    Returns:
        标准化后的文本
    """
    # 移除数字标记 (1., 2., 3., etc.)
    text = re.sub(r"\d+\.\s*", "", text)
    # 移除换行符
    text = text.replace("\n", " ")
    # 移除多余空格并转为小写
    text = " ".join(text.split()).lower().strip()
    return text


def normalize_text_advanced(text: str) -> str:
    """
    高级文本标准化（用于精确匹配，移除停用词和标点）
    Args:
        text: 原始文本
    Returns:
        标准化后的文本
    """
    # 转为小写
    # text = text.lower()

    # 移除标点符号
    text = "".join(ch for ch in text if ch not in string.punctuation)

    # 移除articles (a, an, the)
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # 移除停用词
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]

    # 标准化空格
    text = " ".join(words)

    return text


# ============================================================================
# 评估指标计算模块
# ============================================================================


def compute_f1(prediction: str, ground_truth: str) -> float:
    """计算F1分数"""
    pred_tokens = normalize_text_advanced(prediction).split()
    gt_tokens = normalize_text_advanced(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> int:
    """计算精确匹配分数"""
    return int(normalize_text_advanced(prediction) == normalize_text_advanced(ground_truth))


def compute_accuracy_single(prediction: str, ground_truths: list[str]) -> float:
    """
    计算单个预测的accuracy分数，使用多种匹配策略
    Args:
        prediction: 模型预测结果
        ground_truths: 正确答案列表
    Returns:
        accuracy分数 (0.0 或 1.0)
    """
    # 基础标准化
    norm_pred = normalize_text_advanced(prediction)

    for gt in ground_truths:
        norm_gt = normalize_text_advanced(gt)

        if norm_gt in norm_pred:
            return 1.0

        # pred_words = set(normalize_text_advanced(prediction).split())
        # gt_words = set(normalize_text_advanced(gt).split())

        # if not pred_words or not gt_words:
        #     continue

        # # 如果ground truth的所有关键词都在prediction中
        # if gt_words.issubset(pred_words):
        #     return 1.0

    return 0.0


def evaluate_predictions(
    predictions: list[str], ground_truths: list[list[str]], metric: str = "accuracy"
) -> dict[str, float]:
    """
    评估预测结果
    Args:
        predictions: 预测结果列表
        ground_truths: 正确答案列表的列表
        metric: 评估指标 ("accuracy", "f1", "exact_match", "all")
    Returns:
        评估结果字典
    """
    results: dict[str, float] = {}

    if metric in ["accuracy", "all"]:
        accuracy_scores = [
            compute_accuracy_single(pred, truths)
            for pred, truths in zip(predictions, ground_truths)
        ]
        results["accuracy"] = float(100 * np.mean(accuracy_scores))

    if metric in ["f1", "all"]:
        f1_scores = []
        for pred, truths in zip(predictions, ground_truths):
            # 对每个ground truth计算F1，取最大值
            f1_max = max([compute_f1(pred, gt) for gt in truths]) if truths else 0.0
            f1_scores.append(f1_max)
        results["f1"] = float(100 * np.mean(f1_scores))

    if metric in ["exact_match", "all"]:
        em_scores = []
        for pred, truths in zip(predictions, ground_truths):
            # 对每个ground truth计算EM，取最大值
            em_max = max([compute_exact_match(pred, gt) for gt in truths]) if truths else 0
            em_scores.append(em_max)
        results["exact_match"] = float(100 * np.mean(em_scores))

    return results


def load_results(file_path: str) -> dict[str, Any]:
    """加载推理结果文件"""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def calculate_overall_scores(results_data: dict[str, Any], metric: str = "all") -> dict[str, Any]:
    """
    计算整体评估分数（不输出每个样本的详细分数）

    Args:
        results_data: 推理结果数据
        metric: 评估指标
    Returns:
        包含整体分数的数据
    """
    results = results_data["results"]

    # 提取预测和真实答案
    # 兼容不同的字段名称
    predictions = []
    ground_truths = []

    for item in results:
        # 预测结果字段
        pred = item.get("prediction") or item.get("model_output", "")
        predictions.append(pred)

        # 真实答案字段
        gt = item.get("ground_truth", [])
        if isinstance(gt, str):
            gt = [gt]
        ground_truths.append(gt)

    # 计算整体指标
    overall_scores = evaluate_predictions(predictions, ground_truths, metric)

    # 构建评估结果（只包含整体指标）
    evaluation_result = {
        "metadata": results_data.get("metadata", {}),
        "overall_scores": overall_scores,
        "summary": {
            "total_samples": len(results),
            "evaluation_metric": metric,
        },
    }

    return evaluation_result


def analyze_retrieval_quality(
    evaluation_result: dict[str, Any], results_data: dict[str, Any]
) -> dict[str, Any]:
    """
    分析检索质量

    Args:
        evaluation_result: 评估结果数据
        results_data: 原始结果数据
    Returns:
        检索质量分析结果
    """
    detailed_results = results_data["results"]

    # 统计检索相关信息
    total_samples = len(detailed_results)
    samples_with_context = 0
    context_lengths = []

    # 分析检索上下文与答案的相关性
    context_relevance_scores = []

    for item in detailed_results:
        # 兼容不同的字段名称
        contexts = item.get("retrieved_docs") or item.get("retrieved_context", [])

        if contexts:
            samples_with_context += 1
            context_lengths.append(len(contexts))

            # 简单的相关性分析：检查真实答案是否出现在检索的上下文中
            ground_truth = item.get("ground_truth", [])
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth]

            # 对每个真实答案检查是否在上下文中
            found_in_context = False
            for gt in ground_truth:
                gt_normalized = normalize_text_basic(gt)
                for context in contexts:
                    # 处理不同的上下文格式
                    if isinstance(context, dict):
                        context_text = context.get("text", "")
                    else:
                        context_text = str(context)

                    context_normalized = normalize_text_basic(context_text)
                    if gt_normalized in context_normalized:
                        found_in_context = True
                        break
                if found_in_context:
                    break

            context_relevance_scores.append(1.0 if found_in_context else 0.0)

    retrieval_analysis = {
        "total_samples": total_samples,
        "samples_with_context": samples_with_context,
        "context_coverage": (samples_with_context / total_samples if total_samples > 0 else 0.0),
        "avg_context_count": np.mean(context_lengths) if context_lengths else 0.0,
        "context_relevance_rate": (
            np.mean(context_relevance_scores) if context_relevance_scores else 0.0
        ),
    }

    return retrieval_analysis


def print_evaluation_summary(evaluation_result: dict[str, Any]):
    """打印评估结果摘要"""
    metadata = evaluation_result.get("metadata", {})
    scores = evaluation_result["overall_scores"]
    summary = evaluation_result["summary"]

    print("\n" + "=" * 60)
    print("📊 评估结果摘要")
    print("=" * 60)

    # 打印配置信息
    if metadata:
        print("🔧 配置信息:")
        if "pipeline_name" in metadata:
            print(f"   Pipeline: {metadata['pipeline_name']}")
        if "timestamp" in metadata:
            print(f"   时间: {metadata['timestamp']}")
        if "config" in metadata:
            config = metadata["config"]
            if "pipeline" in config and "pipeline_config" in config["pipeline"]:
                pipeline_config = config["pipeline"]["pipeline_config"]
                if "model_name" in pipeline_config:
                    print(f"   模型: {pipeline_config['model_name']}")
                if "top_k" in pipeline_config:
                    print(f"   Top-K: {pipeline_config['top_k']}")
        if "total_samples" in metadata:
            print(f"   样本数: {metadata['total_samples']}")
        elif "summary" in metadata and "total_samples" in summary:
            print(f"   样本数: {summary['total_samples']}")

    print(f"\n📊 总样本数: {summary['total_samples']}")
    print(f"📏 评估指标: {summary['evaluation_metric']}")

    print("\n📈 整体性能指标:")
    for metric, score in scores.items():
        print(f"   {metric.upper()}: {score:.2f}%")

    # 添加检索质量分析
    if "retrieval_analysis" in evaluation_result:
        retrieval_stats = evaluation_result["retrieval_analysis"]
        print("\n🔍 检索质量分析:")
        print(f"   上下文覆盖率: {100 * retrieval_stats['context_coverage']:.2f}%")
        print(f"   平均检索数量: {retrieval_stats['avg_context_count']:.2f}")
        print(f"   上下文相关性: {100 * retrieval_stats['context_relevance_rate']:.2f}%")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="评估RAG推理结果")
    parser.add_argument("--results", "-r", type=str, required=True, help="推理结果文件路径")
    parser.add_argument(
        "--metric",
        choices=["accuracy", "f1", "exact_match", "all"],
        default="all",
        help="评估指标",
    )
    parser.add_argument("--output", "-o", type=str, help="输出评估结果文件路径")

    args = parser.parse_args()

    # 加载推理结果
    print(f"📥 正在加载推理结果: {args.results}")
    results_data = load_results(args.results)

    # 计算整体评估分数
    print(f"🔄 正在计算评估指标: {args.metric}")
    evaluation_result = calculate_overall_scores(results_data, args.metric)

    # 分析检索质量（如果有检索上下文）
    print("🔍 正在分析检索质量...")
    retrieval_analysis = analyze_retrieval_quality(evaluation_result, results_data)
    evaluation_result["retrieval_analysis"] = retrieval_analysis

    # 打印摘要
    print_evaluation_summary(evaluation_result)

    # 保存评估结果
    if args.output:
        output_path = args.output
    else:
        # 生成默认输出文件名
        input_path = Path(args.results)
        output_filename = f"evaluation_{input_path.stem}.json"
        output_path = get_output_file(output_filename, "benchmarks")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 评估结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
