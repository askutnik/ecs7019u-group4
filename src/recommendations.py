from __future__ import annotations

from typing import Dict, Any, List, Tuple
import json
import streamlit as st
from groq import Groq

#set up and load the groq api key
client = Groq(api_key=st.secrets["GROQ_API_KEY"])


def _call_groq(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Return ONLY valid JSON. No markdown. No explanation."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def _safe_json_parse(text: str) -> Dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("⚠️ INVALID JSON:\n", text)
        return None


#get the necessary ai reccomendation by passing the prompts for the kmeans 
@st.cache_data(show_spinner=True)
def get_ai_recommendation(cluster_id: int, cluster_stats: Dict[str, Any],product_desc: str) -> Dict[str, Any]:

    relevant_stats = {
        k: v for k, v in cluster_stats.items()
        if isinstance(v, (int, float))
    }

    prompt = f"""
Return ONLY valid JSON.

You are a marketing strategist.

PRIMARY OBJECTIVE:
Create a marketing strategy for the PRODUCT below.

PRODUCT:
{product_desc}

TARGET DEMOGRAPHIC (context only, do not just describe it):
{json.dumps(relevant_stats, indent=2)}

IMPORTANT RULES:
- Do NOT just describe the demographic
- ALWAYS tie every insight back to the PRODUCT
- Focus on how to SELL THIS PRODUCT to this audience
- Be specific and practical
IMPORTANT:
- "target_profile" MUST be a single paragraph (string)
- DO NOT return dictionaries or nested objects
- DO NOT separate by cluster
- Blend the audiences into ONE description

Respond in EXACT format:
{{
    "target_profile": "describe audience in relation to the PRODUCT",
    "recommended_channels": ["channel1", "channel2"],
    "messaging_strategy": ["strategy1 focused on product", "strategy2 focused on product"],
    "rationale": "explain WHY this PRODUCT fits this audience using the data"
}}
"""
    try:
        content = _call_groq(prompt)
        print(f"\n[AI Cluster {cluster_id}]\n{prompt}\n")
        print("RAW:", content)

        parsed = _safe_json_parse(content)
        if parsed:
            return parsed

        # retry once
        content = _call_groq(prompt + "\nIMPORTANT: ONLY JSON.")
        parsed = _safe_json_parse(content)

        if parsed:
            return parsed

        return {
            "cluster_name": f"Cluster {cluster_id}",
            "target_profile": "AI failed",
            "recommended_channels": [],
            "messaging_strategy": [],
            "rationale": "Invalid JSON output"
        }

    except Exception as e:
        return {
            "cluster_name": f"Cluster {cluster_id}",
            "target_profile": "Error",
            "recommended_channels": [],
            "messaging_strategy": [],
            "rationale": str(e),
        }


# get the ai reccomendation for the hybrid model
@st.cache_data(show_spinner=True)
def get_ai_recommendation_hybrid(
    cluster_probs: List[Tuple[int, float]],
    cluster_stats_map: Dict[int, Dict[str, Any]],product_desc: str,CLUSTER_LABELS=None
) -> Dict[str, Any]:

    blended: Dict[str, float] = {}

    for cid, weight in cluster_probs:
        stats = cluster_stats_map.get(cid, {})
        for k, v in stats.items():
            if isinstance(v, (int, float)) and k != "cluster":
                blended[k] = blended.get(k, 0.0) + v * weight

    mix_description = ", ".join(
        f"{prob * 100:.0f}% Cluster {cid}" for cid, prob in cluster_probs
    )
    label_map = {cid: CLUSTER_LABELS[cid] for cid, _ in cluster_probs}

    prompt = f"""
Return ONLY valid JSON.

You are a senior marketing strategist.

PRIMARY OBJECTIVE:
Market the PRODUCT below to a MIXED audience.

PRODUCT:
{product_desc}

AUDIENCE MIX:
{mix_description}

Cluster labels (IMPORTANT - use these exact names):
{json.dumps(label_map, indent=2)}

BLENDED DEMOGRAPHIC PROFILE:
{json.dumps({k: round(v, 3) for k, v in blended.items()}, indent=2)}



IMPORTANT RULES:
- Do NOT just describe the audience
- Explain how the PRODUCT appeals to EACH segment
- Resolve differences between segments
- Make strategy actionable
IMPORTANT:
- "target_profile" MUST be a single paragraph (string)
- DO NOT return dictionaries or nested objects
- DO NOT separate by cluster
- Blend the audiences into ONE description
-give the name of the highest probable cluster

Respond in format:
{{
    "target_profile": "audience described in relation to product",
    "recommended_channels": ["channel1", "channel2"],
    "messaging_strategy": ["strategy1 tied to product", "strategy2 tied to product"],
    "rationale": "detailed explanation of product-market fit"
}}
"""

    try:
        content = _call_groq(prompt)
        print(f"\n[AI Hybrid {mix_description}]\n{prompt}\n")
        print("RAW:", content)

        parsed = _safe_json_parse(content)
        if parsed:
            return parsed

        # retry once
        content = _call_groq(prompt + "\nONLY JSON OUTPUT.")
        parsed = _safe_json_parse(content)

        if parsed:
            return parsed

        return {
            "cluster_name": "MUST be EXACTLY one of the provided cluster labels OR a combination like 'Cluster 1 + Cluster 3 Blend'. Do NOT invent semantic names.",
            "target_profile": "AI failed",
            "recommended_channels": [],
            "messaging_strategy": [],
            "rationale": "Invalid JSON from Groq",
        }

    except Exception as e:
        return {
            "cluster_name": "Blended Profile",
            "target_profile": "Error",
            "recommended_channels": [],
            "messaging_strategy": [],
            "rationale": str(e),
        }
