# Databricks notebook source
# MAGIC %pip freeze

# COMMAND ----------

# MAGIC %pip install openai httpx beautifulsoup4

# COMMAND ----------

# MAGIC %pip install mlflow==2.10.1 lxml==4.9.3 transformers==4.30.2 databricks-vectorsearch==0.22 databricks-sdk==0.28.0 databricks-feature-store==0.17.0 langchain==0.2.11 langchain_core==0.2.23
# MAGIC %pip install dspy-ai -U

# COMMAND ----------

# databricksのpythonを再起動させる
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip list

# COMMAND ----------

import os
import openai
import httpx
from bs4.element import Tag
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re

# COMMAND ----------

# 情報収集に使えそうなリンク集
# https://www.tech.ac.jp/sitemap/
# https://www.tech.ac.jp/work_books/
# https://www.tech.ac.jp/blog/

# まだ使えていないページ
# https://content.tech.ac.jp/campuslife/

# PDFいっぱいあるページ
# https://www.tech.ac.jp/info/

# COMMAND ----------

from httpx import Timeout
import time

def get_soup(url):
    with httpx.Client() as client:
        # ReadTimeoutが起きることがあるので、情報取得のタイムアウトを10sにする
        timeout = Timeout(5.0, read=10.0)
        html = client.get(url, timeout=timeout)
    if html.status_code != 200:
        print(f"Failed to get {url}")
        html.raise_for_status()
    return BeautifulSoup(html.content, "html.parser")

# COMMAND ----------

sitemap_url = "https://www.tech.ac.jp/sitemap/"
sitemap_soup = get_soup(sitemap_url)
work_books_url = "https://www.tech.ac.jp/work_books/"
work_books_soup = get_soup(work_books_url)
blog_url = "https://www.tech.ac.jp/blog/"
blog_soup = get_soup(blog_url)

# COMMAND ----------

def extract_links(soup) -> set:
    links = soup.find_all('a')
    links_set = set()
    for link in links:
        url = link.get('href')
        if "http" not in url and url.startswith("/"):
            url = "https://www.tech.ac.jp" + url  # urlが相対パスになっているため、www.~~を追加
            links_set.add(url)

    return links_set

# COMMAND ----------

sitemap_urls = extract_links(sitemap_soup)
work_books_urls = extract_links(work_books_soup)
blog_urls = extract_links(blog_soup)

urls_ls = list(sitemap_urls | work_books_urls | blog_urls)

# COMMAND ----------

# pdfはいったん退避
for url in urls_ls:
  if url.endswith('pdf'):
    print(url)
    urls_ls.remove(url)


# COMMAND ----------

# リンク集は省く
urls_ls.remove(sitemap_url)
urls_ls.remove(work_books_url)
urls_ls.remove(blog_url)

# 学生作品紹介ページ
student_works_url = 'https://www.tech.ac.jp/gallery/'
urls_ls.remove(student_works_url)

# 韓国語、英語、中国語の紹介ページ
lang_urls = [
  'https://www.tech.ac.jp/visitor/language/ko/',
  'https://www.tech.ac.jp/visitor/language/en/',
  'https://www.tech.ac.jp/visitor/language/ch/'
]
for url in lang_urls:
  urls_ls.remove(url)

# 変動するページは一旦やらないようにする
# 理由は、RAGが回答する時期にこのオープンキャンパスがやっているとは限らないから
# もしやるとしたら、イベント用のテーブルを別で作って、月一で更新かけるとかが有効かも
event_urls = [
  'https://www.tech.ac.jp/opencampus/',
  'https://www.tech.ac.jp/opencampus/program/special-event/',
  'https://www.tech.ac.jp/opencampus/program/experience-lesson/',
  'https://www.tech.ac.jp/opencampus/program/exam/',
  'https://www.tech.ac.jp/opencampus/program/school-briefing/',
]
for url in event_urls:
  urls_ls.remove(url)


# COMMAND ----------

# 取得したHPのurl一覧
for url in urls_ls:
    print(url)

# COMMAND ----------

len(urls_ls)

# COMMAND ----------

from typing import Optional

def extract_page_different(soup) -> Tag:
  # main tagがない
  article_html = soup.select("#page > .l-contents > .l-main > article")[0]
  course_info = article_html.select(".p-different_course")
  opencampus_info = article_html.select(".p-different_opencampus")
  article_html.select(".p-different_course")[0].decompose()
  article_html.select(".p-different_opencampus")[0].decompose()
  return article_html

def extract_page_strengths(soup) -> Tag:
  # main tagがない
  return soup.select("#page > .l-contents > .l-main > article")[0]

def extract_page_myschool(soup) -> str:
  # article tagがない
  opencampus_leading = soup.select("#page > main > .p-opencampus_leading")[0]
  myschool_point = soup.select("#page > main > .p-myschool_point")[0]
  return str(opencampus_leading) + str(myschool_point)

def extract_page_webopencampus(soup) -> str:  
  # article tagがない
  opencampus_leading = soup.select("#page > main > .p-opencampus_leading")[0]
  unique_info = soup.select("#page > main > .c-common_section")[0]
  web_opencampus_step = soup.select("#page > main > .c-common_section")[0]
  return str(opencampus_leading) + str(web_opencampus_step)

def remove_tag(article_html, selector):
  if len(article_html.select(selector)) > 0:
    article_html.select(selector)[0].decompose()

def extract_article_html(url) -> Optional[str]:
  soup = get_soup(url)
  try:
    article_html = soup.select("#page > main > article")[0]
    # リンク集
    remove_tag(article_html, ".c-lower_links")
    # OpenCampus情報
    remove_tag(article_html, ".p-course_opencampus")
    remove_tag(article_html, "#opencampus")
    # 専攻一覧
    remove_tag(article_html, ".p-course_major")
    # 専攻リンク
    remove_tag(article_html, ".p-world_links")
    # 資料請求とopemcampus
    remove_tag(article_html, ".c-cta01_sm")
    # workbook内のopemcampus情報
    remove_tag(article_html, ".p-work_books_article__body > .p-work_books__opencampus")
    # パンフレット
    remove_tag(article_html, ".c-admission_cta")

  except:
    # 個別処理
    if url == 'https://www.tech.ac.jp/features/different/':
      article_html = extract_page_different(soup)
    elif url == url == 'https://www.tech.ac.jp/features/strengths/':
      article_html = extract_page_strengths(soup)
    elif url == 'https://www.tech.ac.jp/myschool/':
      article_html = extract_page_myschool(soup)
    elif url == 'https://www.tech.ac.jp/web_opencampus/':
      article_html = extract_page_webopencampus(soup)
    else:
      print('error in url:', url)
      return None

  return str(article_html)

# COMMAND ----------

url_html_pairs = []

import time
from tqdm import tqdm
print('Processing...')
for url in tqdm(urls_ls, desc='Extracting HTML'):
  article_html = extract_article_html(url)
  if article_html is not None:
    url_html_pairs.append({
      'url': url,
      'text': article_html,
    })
  else:
    print('error in url:', url)
  time.sleep(0.5)
print('All done!')

# COMMAND ----------

len(url_html_pairs)

# COMMAND ----------

url_html_pairs[0]

# COMMAND ----------

# 参考
# https://qiita.com/taka_yayoi/items/f174599e4721e51e9e1d
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, OpenAIGPTTokenizer

max_chunk_size = 512

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])

# Split on H2で分割しますが、あまり小さすぎないように小さなh2チャンクはマージします
def split_html_on_h2(html, min_chunk_size = 20, max_chunk_size=512):
  if not html:
      return []
  h2_chunks = html_splitter.split_text(html)
  chunks = []
  previous_chunk = ""
  # チャンクを結合し、h2の前にテキストを追加することでチャンクを結合し、小さすぎる文書を回避します
  for c in h2_chunks:
    # h2の結合 (注意: 重複したh2を回避するために以前のチャンクを削除することもできます)
    content = c.metadata.get('header2', "") + "\n" + c.page_content
    if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size/2:
        previous_chunk += content + "\n"
    else:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))
        previous_chunk = content + "\n"
  if previous_chunk:
      chunks.extend(text_splitter.split_text(previous_chunk.strip()))
  # 小さすぎるチャンクの破棄
  return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]
 

# COMMAND ----------

html = url_html_pairs[0]['text']
split_html = split_html_on_h2(html)

# COMMAND ----------

split_html

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

# sparkですべてのドキュメントのチャンクを作成するためのユーザー定義関数(UDF)を作成
@pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_html_on_h2)
    

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
tmp_raw_data_table_name = f'tmp_{raw_data_table_name}'
sql(f"drop table if exists {tmp_raw_data_table_name}")


spark.createDataFrame(url_html_pairs).write.mode('overwrite').saveAsTable(tmp_raw_data_table_name)

display(spark.table(tmp_raw_data_table_name))

# COMMAND ----------

# すべてのドキュメントチャンクを保存する
(spark.table(tmp_raw_data_table_name)
      .filter('text is not null')
      .withColumn('content', F.explode(parse_and_split('text')))
      .drop("text")
      .write.mode('overwrite').saveAsTable(raw_data_table_name))

sql(f"drop table if exists {tmp_raw_data_table_name}")

display(spark.table(raw_data_table_name))

# COMMAND ----------

sql(f"DROP TABLE IF EXISTS {embed_table_name};")

sql(f"""
--インデックスを作成するには、テーブルのChange Data Feedを有効にします
CREATE TABLE IF NOT EXISTS {embed_table_name} (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY,
  url STRING,
  content STRING
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
# embedding_endpoint_name = "databricks-gte-large-en"
embedding_endpoint_name = "multilingual-e5-large-embedding"

#インデックスを新規作成
print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
vsc.create_delta_sync_index(
  endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
  index_name=vs_index_fullname,
  pipeline_type="TRIGGERED",
  source_table_name=source_table_fullname,
  primary_key="id",
  embedding_source_column="content",
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
    time.sleep(10)
    vs_index.sync()  # なぜかエラー出るが、sync()を2回実行するとエラーが出なくなる


# COMMAND ----------

# インデックスへの参照を取|得
vs_index = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)

# 英語用のモデルを使っているため、回答が良いものではない
# embeddingモデルを日本語用にファインチューニングさせるのもいいかも
  # お家GPUで学習が可能 -> databricksにアップロードする

results = vs_index.similarity_search(
  query_text="授業時間は一コマどのくらいですか？",
  columns=["url", "content"],
  num_results=10,  # 上位三つの結果を返す
)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# RAGチェーンとして呼び方を統一する！

import yaml
import mlflow

rag_chain_config = {
      "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,
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


