# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-vectorsearch langchain==0.2.11 langchain_core==0.2.23 langchain_community==0.2.10 openai

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

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

import time
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

def index_exists(vsc, endpoint_name, index_full_name):
  try:
    dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
    return dict_vsindex.get('status').get('ready', False)
  except Exception as e:
    if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
      print(f'Unexpected error describing the index. This could be a permission issue.')
      raise e
  return False

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

def create_index(vsc, vector_search_index_name):
  # embeddingモデル名
  # embedding_endpoint_name = "databricks-gte-large-en"
  embedding_endpoint_name = "multilingual-e5-large-embedding"
  source_table_name = f"{catalog}.{db}.{embed_table_name}"

  #インデックスを新規作成
  print(f"Creating index {vector_search_index_name} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
  endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
  index_name=vector_search_index_name,
  pipeline_type="TRIGGERED",
  source_table_name=source_table_name,
  primary_key="id",
  embedding_source_column="content",
  embedding_model_endpoint_name=embedding_endpoint_name
  )

  #インデックスの準備ができ、すべてエンベッディングが作成され、インデックスが作成されるのを待ちましょう。
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vector_search_index_name)
  print(f"index {vector_search_index_name} on table {source_table_name} is ready")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
import mlflow

model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

vs_client = VectorSearchClient(disable_notice=True)

if VECTOR_SEARCH_ENDPOINT_NAME not in [e['name'] for e in vs_client.list_endpoints().get('endpoints', [])]:
    vs_client.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vs_client, VECTOR_SEARCH_ENDPOINT_NAME)

if not index_exists(vs_client, VECTOR_SEARCH_ENDPOINT_NAME, model_config.get("vector_search_index_name")):
    create_index(vs_client, model_config.get("vector_search_index_name"))


# COMMAND ----------

#同期をトリガーして、テーブルに保存された新しいデータでベクターサーチのコンテンツを更新
vs_index_fullname = f"{catalog}.{db}.{embed_table_name}_vs_index"

vs_index = vs_client.get_index(
  VECTOR_SEARCH_ENDPOINT_NAME, 
  vs_index_fullname
)

try:
    vs_index.sync()
except Exception as e:
    import time
    time.sleep(10)
    vs_index.sync()  # なぜかエラー出るが、sync()を2回実行するとエラーが出なくなる


# COMMAND ----------

import os

# Specify the full path to the chain notebook
chain_notebook_path = os.path.join(os.getcwd(), "chain_langchain")

# Specify the full path to the config file (.yaml)
config_file_path = os.path.join(os.getcwd(), "rag_chain_config.yaml")

print(f"Chain notebook path: {chain_notebook_path}")
print(f"Chain notebook path: {config_file_path}")

# COMMAND ----------

user_account_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

# Set the experiment name
mlflow.set_experiment(f"/Users/{user_account_name}/rach_rag_experiment")

model_name = f"{catalog}.{dbName}.{registered_model_name}"

# Log the model to MLflow
# TODO: remove example_no_conversion once this papercut is fixed
with mlflow.start_run(run_name="rach_rag_chatbot"):
    # Tag to differentiate from the data pipeline runs
    mlflow.set_tag("type", "chain")

    input_example = {
        "messages": [{"role": "user", "content": "授業時間は一コマどのくらいですか？"}]
    }

    logged_chain_info = mlflow.langchain.log_model(
        lc_model=chain_notebook_path,  # Chain code file e.g., /path/to/the/chain.py
        model_config=config_file_path,  # Chain configuration set in 00_config
        artifact_path="chain",  # Required by MLflow
        input_example=input_example,  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        registered_model_name=model_name,
    )

# COMMAND ----------

chain = mlflow.langchain.load_model(logged_chain_info.model_uri)

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

model_name = f"{catalog}.{dbName}.{registered_model_name}"
uc_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=model_name)

# COMMAND ----------

### Test the registered model
registered_agent = mlflow.langchain.load_model(f"models:/{model_name}/{uc_model_info.version}")

registered_agent.invoke(input_example)

# COMMAND ----------



# COMMAND ----------

# Deploy

import os
import mlflow
from databricks import agents

# modelをdeployする
deployment_info = agents.deploy(
    model_name,
    uc_model_info.version,
    scale_to_zero=True,
)

browser_url = mlflow.utils.databricks_utils.get_browser_hostname()
print(f"\n\nView deployment status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}")

review_instructions = """### Rach FAQチャットボットのテスト手順

チャットボットの品質向上のためにぜひフィードバックを提供ください。

1. **多様な質問をお試しください**：
   - 実際のお客様が尋ねると予想される多様な質問を入力ください。これは、予想される質問を効果的に処理できるか否かを確認するのに役立ちます。

2. **回答に対するフィードバック**：
   - 質問の後、フィードバックウィジェットを使って、チャットボットの回答を評価してください。
   - 回答が間違っていたり、改善すべき点がある場合は、「回答の編集（Edit Response）」で修正してください。皆様の修正により、アプリケーションの精度を向上できます。

3. **回答に付随している参考文献の確認**：
   - 質問に対してシステムから回答される各参考文献をご確認ください。
   - Good👍／Bad👎機能を使って、その文書が質問内容に関連しているかどうかを評価ください。

チャットボットの評価にお時間を割いていただき、ありがとうございます。エンドユーザーに高品質の製品をお届けするためには、皆様のご協力が不可欠です。"""

agents.set_review_instructions(model_name, review_instructions)

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

# Wait for the Review App to be ready
print("\nWaiting for endpoint to deploy.  This can take 15 - 20 minutes.", end="")
w = WorkspaceClient()
now = time.time()
while w.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or w.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    print(".", end="")
    time.sleep(30)

    if time.time() - now > 1800:
        raise Exception("Endpoint did not deploy in 30 minutes")

print(f"\n\nReview App: {deployment_info.review_app_url}")

# COMMAND ----------

from databricks import agents

user_list = ["ttc2350sa0009@edu.tech.ac.jp", "21c1080006ks@edu.tech.ac.jp", "21c1080008st@edu.tech.ac.jp", "hiroshi.ouchiyama@databricks.com"]
agents.set_permissions(model_name=model_name, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

print(f"Share this URL with your stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------


