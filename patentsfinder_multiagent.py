# pip install --upgrade langchain==0.1.14 langgraph==0.0.34 openai google-cloud-bigquery streamlit pandas numpy scikit-learn tiktoken pyyaml

# patentsfinder_multiagent.py

import os
import streamlit as st
import pandas as pd
import numpy as np
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel
from langgraph.graph.message import add_messages
from langgraph.graph.events import EventedGraphRunner
from google.cloud import bigquery
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import SystemMessage, HumanMessage
from typing import TypedDict, List, Optional
import json
import re

# --------------------------
# APIキー認証とLLM初期化
# --------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
gcp_json_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

llm = ChatOpenAI(model="gpt-4", temperature=0.2, openai_api_key=openai_api_key)

# --------------------------
# 状態定義
# --------------------------
class PatentState(TypedDict):
    user_input: str
    tech_views: List[str]
    ipc_candidates: List[str]
    search_params: dict
    patent_df: Optional[pd.DataFrame]
    query_text: Optional[str]
    ranked_df: Optional[pd.DataFrame]
    explanations: Optional[List[str]]
    clarification_needed: Optional[bool]

# --------------------------
# エージェントノード定義
# --------------------------

def theme_tech_agent(state: PatentState):
    """技術観点からの視点抽出"""
    prompt = f"""
    ユーザーが調査したい技術テーマは: "{state['user_input']}"
    技術的な観点で、3点ほど関連するキーワードや関心点を挙げてください。
    """
    resp = llm([HumanMessage(content=prompt)])
    return {"tech_views": [resp.content]}

def theme_market_agent(state: PatentState):
    """市場用途からの視点抽出"""
    prompt = f"""
    ユーザーが調査したいテーマは: "{state['user_input']}"
    市場用途や応用先の視点で、関連トピックを3点ほど挙げてください。
    """
    resp = llm([HumanMessage(content=prompt)])
    return {"tech_views": [resp.content]}

def theme_competitor_agent(state: PatentState):
    """競合技術視点の抽出"""
    prompt = f"""
    ユーザーのテーマ: "{state['user_input']}"
    競合製品や関連する研究・技術の観点から3つほどトピックを挙げてください。
    """
    resp = llm([HumanMessage(content=prompt)])
    return {"tech_views": [resp.content]}

def ipc_classification_agents(state: PatentState):
    """IPCコード分類のクロスレビュー"""
    prompt = f"""
    以下の技術観点に基づいて、関連するIPC分類コードを3〜5件提案してください。\n{state['tech_views']}
    出力形式: IPCコード（例: A01B33/00）とその簡潔な説明
    """
    resp = llm([HumanMessage(content=prompt)])
    codes = re.findall(r"[A-Z]\d{2}[A-Z]?\s*\d{1,2}/\d{1,2}", resp.content)
    return {"ipc_candidates": list(set(code.replace(" ", "") for code in codes))}

def extract_search_params(state: PatentState):
    """条件抽出 + 補完（必要あれば再質問）"""
    prompt = f"""
    ユーザー入力から以下の3項目をJSON形式で抽出してください：
    countries, assignees, publication_from（YYYY-MM-DD形式）。
    入力: "{state['user_input']}"
    """
    resp = llm([HumanMessage(content=prompt)])
    try:
        parsed = json.loads(resp.content.strip())
    except json.JSONDecodeError:
        return {"clarification_needed": True}
    return {"search_params": parsed, "clarification_needed": False}

def clarify_params(state: PatentState):
    """曖昧条件の自動ヒアリング"""
    return {
        "search_params": {
            "countries": ["JP"],
            "assignees": [],
            "publication_from": "2020-01-01"
        },
        "clarification_needed": False
    }

def bq_search_agent(state: PatentState):
    """BigQuery検索"""
    credentials = bigquery.Client.from_service_account_json(gcp_json_path)._credentials
    client = bigquery.Client(credentials=credentials)
    params = state["search_params"]
    ipc_list = [f"'{c}'" for c in state["ipc_candidates"]]
    where = f"ipc.code IN ({','.join(ipc_list)})"
    if "countries" in params:
        where += f" AND country_code IN ({','.join(f'\'{c}\'' for c in params['countries'])})"
    if "assignees" in params and params["assignees"]:
        where += f" AND assignee IN ({','.join(f'\'{a}\'' for a in params['assignees'])})"
    if "publication_from" in params:
        where += f" AND publication_date >= '{params['publication_from']}'"
    sql = f"""
    SELECT
        publication_number,
        (SELECT v.text FROM UNNEST(title_localized) AS v WHERE v.language='en' LIMIT 1) AS title,
        (SELECT v.text FROM UNNEST(abstract_localized) AS v WHERE v.language='en' LIMIT 1) AS abstract,
        publication_date
    FROM `patents-public-data.patents.publications` AS p
        LEFT JOIN UNNEST(p.ipc) AS ipc
    WHERE {where}
    LIMIT 50
    """
    df = client.query(sql).to_dataframe()
    return {"patent_df": df}

def similarity_rank_agent(state: PatentState):
    """Embeddingによる類似度評価"""
    import openai
    openai.api_key = openai_api_key
    abstracts = state["patent_df"]["abstract"].fillna("").tolist()
    query = state["user_input"]
    query_vec = openai.Embedding.create(input=query, model="text-embedding-ada-002")["data"][0]["embedding"]
    patent_vecs = [openai.Embedding.create(input=abs, model="text-embedding-ada-002")["data"][0]["embedding"] for abs in abstracts]
    sims = cosine_similarity([query_vec], patent_vecs)[0]
    df = state["patent_df"].copy()
    df["similarity"] = sims
    df_sorted = df.sort_values("similarity", ascending=False)
    return {"ranked_df": df_sorted}

def explanation_agent(state: PatentState):
    """日本語によるやさしい特許要約"""
    top3 = state["ranked_df"].head(3)
    explanations = []
    for _, row in top3.iterrows():
        prompt = f"以下の特許要約を200字以内の簡単な日本語で解説してください：\n{row['abstract']}"
        resp = llm([HumanMessage(content=prompt)])
        explanations.append(resp.content.strip())
    return {"explanations": explanations}

def orchestrator_agent(state: PatentState):
    """最終出力：統合と表示指示"""
    st.markdown("## 📘 AI特許調査レポート")
    st.markdown("### 🔍 類似特許ランキング")
    st.dataframe(state["ranked_df"][["title", "similarity", "publication_date"]])
    st.markdown("### 📝 日本語要約")
    for i, exp in enumerate(state["explanations"], 1):
        st.markdown(f"**{i}.** {exp}")
    return {}

# --------------------------
# LangGraph グラフ構成
# --------------------------
builder = StateGraph(PatentState)

# テーマ理解：並列視点（技術・市場・競合）
builder.add_node("theme_tech", theme_tech_agent)
builder.add_node("theme_market", theme_market_agent)
builder.add_node("theme_competitor", theme_competitor_agent)
builder.add_node("ipc_classify", ipc_classification_agents)
builder.add_node("extract_params", extract_search_params)
builder.add_node("clarify", clarify_params)
builder.add_node("bq_search", bq_search_agent)
builder.add_node("similarity", similarity_rank_agent)
builder.add_node("explain", explanation_agent)
builder.add_node("orchestrator", orchestrator_agent)

builder.set_entry_point("theme_tech")
builder.add_edge("theme_tech", "theme_market")
builder.add_edge("theme_market", "theme_competitor")
builder.add_edge("theme_competitor", "ipc_classify")
builder.add_edge("ipc_classify", "extract_params")

# 条件の曖昧さによる分岐
builder.add_conditional_edges(
    "extract_params",
    lambda state: "clarify" if state.get("clarification_needed") else "bq_search",
    {"clarify": "clarify", "bq_search": "bq_search"}
)
builder.add_edge("clarify", "bq_search")
builder.add_edge("bq_search", "similarity")
builder.add_edge("similarity", "explain")
builder.add_edge("explain", "orchestrator")
builder.add_edge("orchestrator", END)

graph = builder.compile()

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="🔎 PatentsFinder Multi-Agent", layout="wide")
st.title("🔎 PatentsFinder Multi-Agent by LangGraph")

user_input = st.text_input("調査したい技術内容を入力してください", placeholder="例：逆浸透膜のファウリング抑制技術")

if st.button("調査開始") and user_input:
    initial_state = PatentState(user_input=user_input, tech_views=[], ipc_candidates=[], search_params={})
    graph.invoke(initial_state)
