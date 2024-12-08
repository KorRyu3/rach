# Databricks notebook source
# MAGIC %pip install databricks-agents mlflow mlflow-skinny databricks-vectorsearch

# COMMAND ----------

# MAGIC %pip install langchain==0.2.11 langchain-core==0.2.23 langchain-community==0.2.9

# COMMAND ----------

# databricksのpythonを再起動させる
dbutils.library.restartPython()

# COMMAND ----------

from operator import itemgetter
import mlflow
import os

from databricks.vector_search.client import VectorSearchClient

from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)

## Enable MLflow Tracing
mlflow.langchain.autolog()

from langchain.schema import Document
from typing import Optional, Dict, Any, List

# scoreを返したいので、独自に実装する
# .as_retrieverでやると、similarity_search が内部で呼ばれるため、scoreが返ってくる similarity_search_with_scoreを呼ぶようにしている
# https://qiita.com/Oxyride/items/ac7e32714f5fa673d9e4
class CustomDatabricksVectorSearch(DatabricksVectorSearch):
    # search_type: simirarity
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        *,
        query_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_with_score = self.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            query_type=query_type,
            **kwargs,
        )
        for doc, score in docs_with_score:
            # 類似度スコアを保存する
            doc.metadata['score'] = score
        return [doc for doc, _ in docs_with_score]

    # search_type: similarity_score_threshold
    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """
        Default similarity search with relevance scores. Modify if necessary
        in subclass.
        Return docs and relevance scores in the range [0, 1].

        0 is dissimilar, 1 is most similar.

        Args:
            query: Input text.
            k: Number of Documents to return. Defaults to 4.
            **kwargs: kwargs to be passed to similarity search. Should include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of Tuples of (doc, similarity_score)
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        docs_and_similarity_scores = [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]
        for doc, score in docs_and_similarity_scores:
            # 類似度スコアを保存する
            doc.metadata['score'] = score
        return docs_and_similarity_scores


############
# Helper functions
############
# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]


# FIT AND FINISH: We should not require a value here.
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

############
# Connect to the Vector Search Index
############
vs_client = VectorSearchClient(disable_notice=True)

vs_index = vs_client.get_index(
    endpoint_name=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("vector_search_index_name")
)

############
# Turn the Vector Search index into a LangChain retriever
############
vector_search_as_retriever = CustomDatabricksVectorSearch(
    vs_index,
    text_column="content",
    columns=[
        "id",
        "content",
        "url",
    ],
# ).as_retriever(search_kwargs={
#     "k": 10,
#     "query_type": "ann",
# })
).as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        'score_threshold': 0.7,
        # 'query_type': 'ann'  # 0.62s -> 0.6s
    }
)

############
# Required to:
# 1. Enable the RAG Studio Review App to properly display retrieved chunks
# 2. Enable evaluation suite to measure the retriever
############

mlflow.models.set_retriever_schema(
    primary_key="id",
    text_column="content",
    doc_uri="url",  # Review App uses `doc_uri` to display chunks from the same document in a single view
    other_columns=["score"],
)


############
# Method to format the docs returned by the retriever into the prompt
############
def format_context(docs):
    chunk_template = "Passage: {chunk_text}\n"
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
            document_uri=d.metadata["url"],
            score=d.metadata["score"],
        )
        for d in docs
    ]

    return "".join(chunk_contents)

# COMMAND ----------

# vector_search_as_retriever.invoke("授業時間は一コマどのくらいですか？")

# COMMAND ----------

############
# Prompt Template for generation
############
prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions
            "system",
            "【参考情報】のみを参考にしながら【質問】にできるだけ正確に答えてください。わからない場合や、質問が適切でない場合は、分からない旨を答えてください。【参考情報】に記載されていない事実を答えるのはやめてください。",
        ),
        # User's question
        ("user", """【参考情報】
{context}

【質問】
{question}"""),
    ]
)


# COMMAND ----------

############
# FM for generation
############
model = ChatDatabricks(
    endpoint=model_config.get("llm_endpoint_name"),
    extra_params={"temperature": 0.01, "max_tokens": 1500},
)

############
# RAG Chain
############
chain = (
    {
        # userの質問
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        # 参考情報
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)

## Tell MLflow logging where to find your chain.
# `mlflow.models.set_model(model=...)` function specifies the LangChain chain to use for evaluation and deployment.  This is required to log this chain to MLflow with `mlflow.langchain.log_model(...)`.

mlflow.models.set_model(model=chain)

# COMMAND ----------

# input_example = {
#   "messages": [{"role": "user", "content": "授業時間は一コマどのくらいですか？"}]
# }

# chain.invoke(input_example)

# COMMAND ----------


