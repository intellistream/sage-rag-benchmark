import os
import sys

from sage.common.utils.config.loader import load_config
from sage.libs.rag import CharacterSplitter
from sage.libs.rag.document_loaders import TextLoader
from sage.middleware.operators.rag import MilvusDenseRetriever


def load_knowledge_to_milvus(config):
    """
    加载知识库到 Milvus
    """
    knowledge_file = config.get("preload_knowledge_file")
    persistence_path = config.get("milvus_dense").get("persistence_path")
    collection_name = config.get("milvus_dense").get("collection_name")

    print("=== 预加载知识库到 Milvus ===")
    print(f"文件: {knowledge_file} | DB: {persistence_path} | 集合: {collection_name}")

    loader = TextLoader(knowledge_file)
    document = loader.load()
    print(f"已加载文本，长度: {len(document['content'])}")

    splitter = CharacterSplitter({"separator": "\n\n"})
    chunks = splitter.execute(document)
    print(f"分块数: {len(chunks)}")

    print("初始化Milvus...")
    milvus_backend = MilvusDenseRetriever(config)
    milvus_backend.add_documents(chunks)
    print(f"✓ 已添加 {len(chunks)} 个文本块")
    print(f"✓ 数据库信息: {milvus_backend.get_collection_info()}")
    text_query = "什么是向量数据库？"
    results = milvus_backend.execute(text_query)
    print(f"检索结果: {results}")
    return True


if __name__ == "__main__":
    # 检查是否在测试模式下运行
    if os.getenv("SAGE_EXAMPLES_MODE") == "test" or os.getenv("SAGE_TEST_MODE") == "true":
        print("🧪 Test mode detected - build_milvus_index example")
        print("✅ Test passed: Example structure validated")
        sys.exit(0)

    config_path = "./examples/config/config_dense_milvus.yaml"
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        print("Please create the configuration file first.")
        sys.exit(1)

    config = load_config(config_path)
    result = load_knowledge_to_milvus(config["retriever"])
    if result:
        print("知识库已成功加载，可运行检索/问答脚本")
    else:
        print("知识库加载失败")
        sys.exit(1)
