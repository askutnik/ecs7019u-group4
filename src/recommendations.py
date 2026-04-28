from __future__ import annotations
from typing import Dict,Any
import google.generativeai as genai
import streamlit as st
import os
import json


genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

@st.cache_data(show_spinner=True)
def get_ai_recommendation(cluster_id: int, cluster_stats: Dict[str, Any]) -> Dict[str, Any]:
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Filter only relevant stats for the prompt
    relevant_stats = {k: v for k, v in cluster_stats.items() if isinstance(v, (int, float))}
    
    prompt = f"""
    You are an expert marketing strategist.
    Analyze the following demographic data for a specific cluster:
    {json.dumps(relevant_stats, indent=2)}
    
    Based ONLY on these statistics, provide a structured marketing recommendation in JSON format:
    {{
        "cluster_name": "Short descriptive title",
        "target_profile": "Who is this demographic?",
        "recommended_channels": ["Channel 1", "Channel 2"],
        "messaging_strategy": ["Point 1", "Point 2"],
        "rationale": "Reasoning based on the provided stats."
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        content = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        # ADD THIS PRINT STATEMENT
        print(f"AI ERROR: {e}") 
        # Return a more descriptive fallback
        return {
            "cluster_name": f"Cluster {cluster_id}",
            "target_profile": "Error: AI generation failed.",
            "recommended_channels": ["N/A"],
            "messaging_strategy": ["Check terminal for errors."],
            "rationale": f"API Error: {str(e)}"
        }