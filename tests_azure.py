import os
from typing import List
import json
import argparse
import logging

from src.hipporag import HippoRAG


def main():
    parser = argparse.ArgumentParser(description="Testing HippoRAG")
    parser.add_argument('--azure_endpoint', type=str, default=None, help='Azure Endpoint URL')
    parser.add_argument('--azure_embedding_endpoint', type=str, default=None, help='Azure Embedding Endpoint')
    args = parser.parse_args()

    # ========== 重要：设置 Azure OpenAI 凭证 ==========
    os.environ["AZURE_OPENAI_API_KEY"] = "API Key"
    # 也可以设置通用的 OPENAI_API_KEY，有些代码可能使用这个
    os.environ["OPENAI_API_KEY"] = "API Key"

    # 设置 Azure OpenAI 特定的环境变量
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://genai-jp.openai.azure.com/"
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"
    # ================================================

    azure_endpoint = args.azure_endpoint or "https://genai-jp.openai.azure.com/openai/deployments/ln-gpt40/chat/completions?api-version=2024-02-15-preview"

    azure_embedding_endpoint = args.azure_embedding_endpoint or "https://genai-jp.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"

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

    save_dir = 'outputs/azure_test'  # Define save directory for HippoRAG objects (each LLM/Embedding model combination will create a new subdirectory)
    llm_model_name = 'gpt-4o-mini'  # Any OpenAI model name
    # embedding_model_name = 'text-embedding-3-small'  # Embedding model name (NV-Embed, GritLM or Contriever for now)
    #
    # # Startup a HippoRAG instance
    # hipporag = HippoRAG(save_dir=save_dir,
    #                     llm_model_name=llm_model_name,
    #                     embedding_model_name=embedding_model_name,
    #                     azure_endpoint=azure_endpoint,
    #                     azure_embedding_endpoint=azure_embedding_endpoint
    #                     )

    embedding_model_name = 'facebook/contriever'  # 使用Contriever作为嵌入模型

    # 初始化HippoRAG，只传递Azure的LLM端点，不传递嵌入模型端点（因为使用本地模型）
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        azure_endpoint=azure_endpoint)

    # Run indexing
    hipporag.index(docs=docs)

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

    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers)[-2:])

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        azure_endpoint="https://bernal-hipporag.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
                        azure_embedding_endpoint="https://bernal-hipporag.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
                        )

    print(hipporag.rag_qa(queries=queries,
                                  gold_docs=gold_docs,
                                  gold_answers=answers)[-2:])

    # Startup a HippoRAG instance
    hipporag = HippoRAG(save_dir=save_dir,
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        azure_endpoint="https://bernal-hipporag.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
                        azure_embedding_endpoint="https://bernal-hipporag.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
                        )

    new_docs = [
        "Tom Hort's birthplace is Montebello.",
        "Sam Hort's birthplace is Montebello.",
        "Bill Hort's birthplace is Montebello.",
        "Cam Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County.."]

    # Run indexing
    hipporag.index(docs=new_docs)

    print(hipporag.rag_qa(queries=queries,
                          gold_docs=gold_docs,
                          gold_answers=answers)[-2:])

    docs_to_delete = [
        "Tom Hort's birthplace is Montebello.",
        "Sam Hort's birthplace is Montebello.",
        "Bill Hort's birthplace is Montebello.",
        "Cam Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County.."
    ]

    hipporag.delete(docs_to_delete)

    print(hipporag.rag_qa(queries=queries,
                          gold_docs=gold_docs,
                          gold_answers=answers)[-2:])

if __name__ == "__main__":
    main()
