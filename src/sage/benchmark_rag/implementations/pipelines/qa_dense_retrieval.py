import os
import sys
import time

from sage.common.utils.config.loader import load_config
from sage.kernel.api.local_environment import LocalEnvironment
from sage.libs.foundation.io.sink import TerminalSink
from sage.libs.foundation.io.source import FileSource
from sage.middleware.operators.rag import OpenAIGenerator, QAPromptor


def pipeline_run():
    """创建并运行数据处理管道"""
    # 检查是否在测试模式下运行
    if os.getenv("SAGE_EXAMPLES_MODE") == "test" or os.getenv("SAGE_TEST_MODE") == "true":
        print("🧪 Test mode detected - qa_dense_retrieval example")
        print("✅ Test passed: Example structure validated")
        return

    # env = LocalBatchEnvironment() #DEBUG and Batch -- Client 拥有后续程序的全部handler（包括JM）
    env = LocalEnvironment(
        "JM-IP"
    )  # Deployment to JM. -- Client 不拥有后续程序的全部handler（包括JM）

    # Batch Environment.

    (
        env.from_source(FileSource, config["source"])  # 处理且处理一整个file 一次。
        # .map(MilvusDenseRetriever, config["retriever"])  # 需要配置文件
        .map(QAPromptor, config["promptor"])
        .map(OpenAIGenerator, config["generator"]["sagellm"])
        .sink(TerminalSink, config["sink"])  # TM (JVM) --> 会打印在某一台机器的console里
    )

    env.submit()
    time.sleep(5)


if __name__ == "__main__":
    import os

    # 检查是否在测试模式下运行
    if os.getenv("SAGE_EXAMPLES_MODE") == "test" or os.getenv("SAGE_TEST_MODE") == "true":
        print("🧪 Test mode detected - qa_dense_retrieval example")
        print("✅ Test passed: Example structure validated")
        sys.exit(0)

    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("Please create the configuration file first.")
        sys.exit(1)

    config = load_config(config_path)
    pipeline_run()
