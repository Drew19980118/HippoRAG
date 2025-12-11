import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG

def main():
    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]

    # 关键修改 1: 更新模型名和基础URL
    save_dir = 'outputs/local_test_phi'  # 建议使用新目录，避免与旧索引冲突
    llm_model_name = 'microsoft/Phi-3.5-mini-instruct'  # 与vLLM服务加载的模型ID一致
    embedding_model_name = 'nvidia/NV-Embed-v2'
    llm_base_url = "http://localhost:8000/v1"  # 端口改为你实际使用的8000

    # Startup a HippoRAG instance
    print(f"正在初始化HippoRAG，连接本地模型: {llm_model_name} ...")
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        llm_base_url=llm_base_url
                        )

    # Run indexing
    print("开始索引文档...")
    hipporag.index(docs=docs)
    print("索引完成！")

    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    # For Evaluation
    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
         "The prince used the lost glass slipper to search the kingdom.",
         "When the slipper fit perfectly, Cinderella was reunited with the prince."],
        ["Erik Hort's birthplace is Montebello.",
         "Montebello is a part of Rockland County."]
    ]

    print("\n--- 开始本地模型问答测试 ---")
    results = hipporag.rag_qa(queries=queries,
                              gold_docs=gold_docs,
                              gold_answers=answers)

    # 打印最后两个结果（通常是评估指标和答案列表）
    print(results[-2:])

    # 以下部分为测试其他配置（如Azure），已注释掉，确保先集中测试本地服务。
    # 如需测试，请取消注释并确保Azure配置正确。
    """
    print("\n--- 开始Azure OpenAI测试 (需确保配置正确) ---")
    hipporag_azure = HippoRAG(save_dir=save_dir + '_azure',
                              llm_model_name='gpt-4o-mini', # Azure部署名
                              embedding_model_name='text-embedding-3-small',
                              azure_endpoint="https://bernal-hipporag.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
                              azure_embedding_endpoint="https://bernal-hipporag.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
                              )
    results_azure = hipporag_azure.rag_qa(queries=queries,
                                          gold_docs=gold_docs,
                                          gold_answers=answers)
    print(results_azure[-2:])
    """

if __name__ == "__main__":
    main()