# patentsfinder_multiagent.py

# pip install -r requirements.txt

import os
import json
import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2 import service_account
from google.cloud import bigquery

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

# ------------------------------
# 認証（OpenAI / GCP）
# ------------------------------
st.set_page_config(page_title="🔎 PatentsFinder Multi-Agent", layout="wide")
st.title("🔎 PatentsFinder Multi-Agent by LangGraph")

openai_api_key = st.text_input("OpenAI API Key", type="password")
openai_auth_ok = False
if openai_api_key:
    try:
        import openai
        openai.api_key = openai_api_key
        openai.embeddings.create(input="test", model="text-embedding-ada-002")
        st.success("OpenAI APIキーの認証に成功しました。")
        openai_auth_ok = True
    except Exception as e:
        st.error(f"OpenAI APIキーの認証に失敗しました: {e}")
else:
    st.stop()

gcp_json_str = st.text_area("Google Cloud サービスアカウントキー（JSONを貼り付け）", height=200)
gcp_auth_ok = False
if gcp_json_str:
    try:
        gcp_info = json.loads(gcp_json_str)
        GCP_CREDENTIALS = service_account.Credentials.from_service_account_info(gcp_info)
        BQ_PROJECT = gcp_info.get("project_id")
        client = bigquery.Client(project=BQ_PROJECT, credentials=GCP_CREDENTIALS)
        client.query("SELECT 1").result()
        st.success("Google Cloud サービスアカウント認証に成功しました。")
        gcp_auth_ok = True
    except Exception as e:
        st.error(f"Google Cloud サービスアカウント認証に失敗しました: {e}")
else:
    st.stop()

if not (openai_auth_ok and gcp_auth_ok):
    st.stop()

# ------------------------------
# LangGraph ノード定義と実行
# ------------------------------

EMBEDDING_MODEL = "text-embedding-ada-002"
BQ_PUBLIC_PROJECT = "patents-public-data"
BQ_DATASET = "patents"
BQ_TABLE = "publications"
BQ_LOCATION = "US"
BQ_LIMIT = 100

from langgraph.graph import StateGraph

class PatentState(TypedDict):
    user_input: str
    ipc_codes: List[str]
    search_params: dict
    patent_df: Optional[pd.DataFrame]
    query_text: Optional[str]
    ranked_df: Optional[pd.DataFrame]
    explanations: Optional[List[str]]

llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=openai_api_key)

# --- 各エージェントノード（関数） ---
def ipc_classification_node(state: PatentState):
    prompt = f"""
    以下の技術内容に関連するIPCコードを3~5個、簡潔な説明とともに提案してください。
    内容: {state['user_input']}
    """
    res = llm([HumanMessage(content=prompt)])
    codes = re.findall(r"[A-Z]\d{2}[A-Z]?\s*\d{1,2}/\d{1,2}", res.content)
    return {"ipc_codes": list(set(code.replace(" ", "") for code in codes))}

def extract_search_params_node(state: PatentState):
    prompt = f"""
    次のユーザー入力から、countries, assignees, publication_from を含むJSON形式の検索条件を生成してください。
    入力: {state['user_input']}
    """
    res = llm([HumanMessage(content=prompt)])
    try:
        parsed = json.loads(res.content.strip())
    except json.JSONDecodeError:
        parsed = {"countries": ["JP"], "assignees": [], "publication_from": "2020-01-01"}
    return {"search_params": parsed}

def bq_search_node(state: PatentState):
    client = bigquery.Client(project=BQ_PROJECT, credentials=GCP_CREDENTIALS, location=BQ_LOCATION)
    where = []
    ipc = [f"'{c}'" for c in state["ipc_codes"]]
    where.append(f"ipc.code IN ({','.join(ipc)})")
    sp = state["search_params"]
    if sp.get("countries"):
        where.append(f"country_code IN ({','.join([f'\'{c}\'' for c in sp['countries']])})")
    if sp.get("assignees"):
        where.append(f"assignee IN ({','.join([f'\'{a}\'' for a in sp['assignees']])})")
    if sp.get("publication_from"):
        where.append(f"publication_date >= '{sp['publication_from']}'")
    where_clause = " AND ".join(where)
    sql = f"""
    SELECT publication_number,
           (SELECT text FROM UNNEST(title_localized) WHERE language='en' LIMIT 1) AS title,
           (SELECT text FROM UNNEST(abstract_localized) WHERE language='en' LIMIT 1) AS abstract,
           publication_date
    FROM `{BQ_PUBLIC_PROJECT}.{BQ_DATASET}.{BQ_TABLE}` AS p
    LEFT JOIN UNNEST(p.ipc) AS ipc
    WHERE {where_clause}
    LIMIT {BQ_LIMIT}
    """
    df = client.query(sql).to_dataframe()
    return {"patent_df": df}

def similarity_node(state: PatentState):
    import openai
    openai.api_key = openai_api_key
    query_vec = openai.embeddings.create(input=state["user_input"], model=EMBEDDING_MODEL)["data"][0]["embedding"]
    texts = state["patent_df"]["abstract"].fillna("").tolist()
    vecs = [openai.embeddings.create(input=txt, model=EMBEDDING_MODEL)["data"][0]["embedding"] for txt in texts]
    sims = cosine_similarity([query_vec], vecs)[0]
    df = state["patent_df"].copy()
    df["similarity"] = sims
    return {"ranked_df": df.sort_values("similarity", ascending=False)}

def explain_node(state: PatentState):
    top = state["ranked_df"].head(3)
    results = []
    for _, row in top.iterrows():
        msg = f"以下の要約を200字で簡潔に日本語で解説してください:\n{row['abstract']}"
        res = llm([HumanMessage(content=msg)])
        results.append(res.content.strip())
    return {"explanations": results}

def final_display_node(state: PatentState):
    st.markdown("## 🔍 特許検索結果")
    st.dataframe(state["ranked_df"][["title", "similarity"]])
    st.markdown("## 📝 上位3件の解説")
    for i, text in enumerate(state["explanations"], 1):
        st.markdown(f"**{i}.** {text}")
    return {}

# --- LangGraph定義 ---
workflow = StateGraph(PatentState)
workflow.add_node("classify_ipc", ipc_classification_node)
workflow.add_node("extract_params", extract_search_params_node)
workflow.add_node("search_bq", bq_search_node)
workflow.add_node("similarity", similarity_node)
workflow.add_node("explain", explain_node)
workflow.add_node("final", final_display_node)
workflow.set_entry_point("classify_ipc")
workflow.add_edge("classify_ipc", "extract_params")
workflow.add_edge("extract_params", "search_bq")
workflow.add_edge("search_bq", "similarity")
workflow.add_edge("similarity", "explain")
workflow.add_edge("explain", "final")
workflow.add_edge("final", END)

# --- 実行 ---
if user_input := st.text_input("技術内容を入力してください", placeholder="例: AIを用いた膜汚染検知"):
    graph = workflow.compile()
    graph.invoke({"user_input": user_input})
