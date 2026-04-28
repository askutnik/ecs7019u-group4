from __future__ import annotations
from typing import Dict, List

def get_cluster_recommendation(cluster_id:int)->Dict[str,object]:
    # returns marketing recommendations and explanations for a given cluster
    if cluster_id == 0:
        return {
            "cluster_name":"Economically Active Professionals",
            "target_profile":"Economically active, working-age professionals with stable income and strong purchasing power",
            "recommended_channels":["LinkedIn","Google Ads","Meta Ads","Email marketing"],
            "messaging_strategy":["Emphasise quality and convenience","Highlight brand credibility and reviews","Time saving and premium value suggestions"],
            "rationale":("This cluster exhibits high economic activity, low unemployment and above average self-employment, indicating strong purchasing power and responsiveness to professional and lifestyle oriented marketing.")
        }

    if cluster_id == 1:
        return {
            "cluster_name":"Student Focused Urban Areas",
            "target_profile":"Digital focused, price sensitive student populations in urban areas with high student proportions and low long-term economic stability",
            "recommended_channels":["TikTok","Instagram","Event sponsorships at universities","Exclusive student discounts","Partnerships with student organisations"],
            "messaging_strategy":["Price sensitive offers and discounts","Short form visually engaging content","Limited-time promotions"],
            "rationale":("This cluster shows a very high student proportion and low long-term economic stability. Younger audiences in these areas are highly digitally engaged and respond best to fast, low-commitment marketing formats.")
        }

    if cluster_id == 2:
        return {
            "cluster_name":"Older Rural Communities",
            "target_profile":"Older populations with high retirement and economic inactivity",
            "recommended_channels":["Facebook","Local newspapers","Mail campaigns","Community events and sponsorships","Search enginge optimization"],
            "messaging_strategy":["Trust, reliability and familiarity","Clear and simple messaging","Community-oriented values"],
            "rationale":("This cluster has a high proportion of retired individuals and lower economic activity. Marketing strategies should prioritise trust and clarity over trend-driven messaging by using established and familiar channels.")
        }

    if cluster_id == 3:
        return {
            "cluster_name":"Economically Disadvantaged Areas",
            "target_profile":"Areas experiencing higher deprivation and economic inactivity",
            "recommended_channels":["Facebook","SMS ads","Community notice boards","Local outreach programmes"],
            "messaging_strategy":["Value focused and affordable offerings","Avoid luxurious framing","Inclusive language"],
            "rationale":("This cluster exhibits high deprivation, unemployment and health-related inactivity. Marketing in these areas should be ethically sensitive, avoiding exploitative practices and focusing on accessibility and genuine value.")
        }

    # fallback
    return {
        "cluster_name":"unknown",
        "target_profile":"No profile available",
        "recommended_channels":[],
        "messaging_strategy":[],
        "rationale":"No recommendation available for this cluster."
    }
