# Databricks notebook source
# MAGIC %pip freeze

# COMMAND ----------

# MAGIC %pip install openai httpx beautifulsoup4

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 transformers==4.30.2 databricks-vectorsearch==0.22 databricks-sdk==0.28.0 databricks-feature-store==0.17.0
# MAGIC %pip install dspy-ai -U

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# databricksのpythonを再起動させる
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip list

# COMMAND ----------

import os
import openai
import httpx
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re

# COMMAND ----------

load_url = "https://www.tech.ac.jp/sitemap/"
html = httpx.get(load_url)
soup = BeautifulSoup(html.content, "html.parser")

# COMMAND ----------

links = soup.find_all('a')
links_set = set()
for link in links:
    url = link.get('href')
    if "http" not in url and url.startswith("/"):
        url = "https://www.tech.ac.jp" + url  # urlが相対パスになっているため、www.~~を追加
        links_set.add(url)

urls_ls = list(links_set)

# COMMAND ----------

# sitemapから取ってきた、HPのurl一覧
urls_ls

# COMMAND ----------

q_and_a = "https://www.tech.ac.jp/school/faq/"

html = httpx.get(q_and_a)
soup = BeautifulSoup(html.content, "html.parser")

# COMMAND ----------

qa_list = []
qa_selector = soup.select("#page > main > article > #faq01,#faq02,#faq03")

for faq_container in qa_selector:
    faq_item = faq_container.select("div > div > div > ul > li")
    
    for faq in faq_item:
        q = faq.select(".-q > p")[0].text
        a = faq.select(".-a > p")[0].text
        print(q)
        print(a)
        print("-"*40)

        qa_list.append({
            "query": q,
            "response": a
        })

# COMMAND ----------


len(qa_list)

# COMMAND ----------

qa_list

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf
import os
import mlflow
from mlflow import MlflowClient

# COMMAND ----------

[r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]

# COMMAND ----------

def use_and_create_db(catalog, dbName, cloud_storage_path = None):
  print(f"USE CATALOG `{catalog}`")
  spark.sql(f"USE CATALOG `{catalog}`")
  spark.sql(f"""create database if not exists `{dbName}` """)

assert catalog not in ['hive_metastore', 'spark_catalog']
#If the catalog is defined, we force it to the given value and throw exception if not.
if len(catalog) > 0:
  current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
  if current_catalog != catalog:
    catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
    if catalog not in catalogs:
      # spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
      if catalog == 'dbdemos':
        spark.sql(f"ALTER CATALOG {catalog} OWNER TO `account users`")
  use_and_create_db(catalog, dbName)

# COMMAND ----------

[r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]

# COMMAND ----------

# sql(f"CREATE CATALOG IF NOT EXISTS {catalog};")
sql(f"USE CATALOG {catalog};")
sql(f"CREATE SCHEMA IF NOT EXISTS {dbName};")
sql(f"USE SCHEMA {dbName};")
sql(f"CREATE VOLUME IF NOT EXISTS {volume};")

# COMMAND ----------

# すでに同名のテーブルが存在する場合は削除
sql(f"drop table if exists {raw_data_table_name}")


spark.createDataFrame(qa_list).write.mode('overwrite').saveAsTable(raw_data_table_name)

display(spark.table(raw_data_table_name))

# COMMAND ----------

sql(f"DROP TABLE IF EXISTS {embed_table_name};")

sql(f"""
--インデックスを作成するには、テーブルのChange Data Feedを有効にします
CREATE TABLE IF NOT EXISTS {embed_table_name} (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY,
  query STRING,
  response STRING
) TBLPROPERTIES (delta.enableChangeDataFeed = true); 
""")

spark.table(raw_data_table_name).write.mode('overwrite').saveAsTable(embed_table_name)

display(spark.table(embed_table_name))

# COMMAND ----------

import time

def index_exists(vsc, endpoint_name, index_full_name):
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get('status').get('ready', False)
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False
  

def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    endpoint = vsc.get_endpoint(vs_endpoint_name)
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# エンドポイントは自分で立ち上げる
# エンドポイントは毎回立ち上げるもの -> ずっと起動していると、お金がかかっちゃう
# エンドポイントを落とすと、紐づいているベクトルインデックスが全て削除されちゃう
    # 毎度起動する際は、ベクトルインデックスも作らなければならない
vsc = VectorSearchClient()

if VECTOR_SEARCH_ENDPOINT_NAME not in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c
import time

#インデックスの元となるテーブル
source_table_fullname = f"{catalog}.{db}.{embed_table_name}"

#インデックスを格納する場所
vs_index_fullname = f"{catalog}.{db}.{embed_table_name}_vs_index"

#すでに同名のインデックスが存在すれば削除
if index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Deleting index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.delete_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  while True:
    if index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
      time.sleep(1)
      print(".")
    else:      
      break

# embeddingモデル名
embedding_endpoint_name = "databricks-gte-large-en"

#インデックスを新規作成
print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
vsc.create_delta_sync_index(
  endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
  index_name=vs_index_fullname,
  pipeline_type="TRIGGERED",
  source_table_name=source_table_fullname,
  primary_key="id",
  embedding_source_column="response",
  embedding_model_endpoint_name=embedding_endpoint_name
)

#インデックスの準備ができ、すべてエンベッディングが作成され、インデックスが作成されるのを待ちましょう。
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

#同期をトリガーして、テーブルに保存された新しいデータでベクターサーチのコンテンツを更新
vs_index = vsc.get_index(
  VECTOR_SEARCH_ENDPOINT_NAME, 
  vs_index_fullname)

try:
    vs_index.sync()
except Exception as e:
    import time
    time.sleep(5)
    vs_index.sync()  # なぜかエラー出るが、sync()を2回実行するとエラーが出なくなる


# COMMAND ----------



# COMMAND ----------

# インデックスへの参照を取|得
vs_index = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

# 英語用のモデルを使っているため、回答が良いものではない
# embeddingモデルを日本語用にファインチューニングさせるのもいいかも
  # お家GPUで学習が可能 -> databricksにアップロードする

results = vs_index.similarity_search(
  query_text="授業時間はどのくらいですか？",
  columns=["query", "response"],
  num_results=10  # 上位三つの結果を返す
)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# RAGチェーンとして呼び方を統一する！

import yaml
import mlflow

rag_chain_config = {
      "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,
      "source_table_name": f"{catalog}.{dbName}.{embed_table_name}",
      "vector_search_index_name": f"{catalog}.{dbName}.{embed_table_name}_vs_index",
      "llm_endpoint_name": instruct_endpoint_name,
}
config_file_name = 'rag_chain_config.yaml'
try:
    with open(config_file_name, 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')

# COMMAND ----------

import os

API_ROOT = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ["DATABRICKS_HOST"] = API_ROOT
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DATABRICKS_TOKEN"] = API_TOKEN

# COMMAND ----------


