# making the main orchestrator that runs the whole agentic loop

import pandas as pd
import numpy as np
import os
 
from agents.demand_agent import DemandAgent
from agents.pricing_agent import PricingAgent
from agents.monitor_agent import MonitorAgent

#load the processed datasets

urban=pd.read_csv("final_datasets\\aligned_urbanev_data.csv")

acn=pd.read_csv("final_datasets\\acn_processed.csv")

# Verify that required columns exist before running anything
urban_required = ["gridID", "timestamp", "hour", "day_of_week", "is_weekend", "is_peak_hr","CBD", "dynamic_pricing", "count", "fast_count","rolling_3h_volume", "occupancy_density", "queue_length_proxy","charger_util_rate", "volume"]

acn_required = ["hour", "session_duration_hr","charging_duration_hr", "is_peak_hour","is_weekend", "charger_util_rate", "queue_length_proxy","idle_time_hr", "day_of_week", "revenue_per_session", "kWhDelivered"]

missing_urban = [c for c in urban_required if c not in urban.columns]
missing_acn = [c for c in acn_required if c not in acn.columns]

if missing_urban:
    print(f"WARNING: urban_processed.csv missing columns:{missing_urban}")
if missing_acn:
    print(f"WARNING: acn_processed.csv missing columns:{missing_acn}")

#initialise the agents

demand_agent= DemandAgent()
pricing_agent = PricingAgent()
monitor_agent = MonitorAgent(pricing_agent)

#train the revenue model once before the episode loop so that pricing agent can use revnue by hour
pricing_agent.train_revenue_model(acn)


# now create the episode loop
N_EPISODES = 15
all_metrics = []

# to enhance learning make equal sized time chunks so that each episode sees different data
urban['datetime'] = pd.to_datetime(urban['timestamp'])
urban = urban.sort_values('datetime')
chunk_size = len(urban) // N_EPISODES

# Train demand model once on full dataset before episode loop starts without counting as an episode
demand_agent.observe(urban)
demand_agent.think()

for episode in range(1, N_EPISODES + 1):
    print(f"EPISODE {episode} / {N_EPISODES}")


    episode_urban = urban.iloc[(episode - 1) * chunk_size : episode * chunk_size].copy()
    # episode_urban = urban.sample(frac=0.6, random_state=episode).copy()

    #run the demand agent
    print(f"\n DEMAND PREDICTION AGENT")
    print(f"Input: UrbanEV district-level data")
    print(f"Output : predicted_util per gridID × hour (DemandSignal)")
 
    demand_signal = demand_agent.run_episode(episode_urban)

    print(f"DemandSignal : {len(demand_signal)} district-hour rows")
    print(f"Mean predicted util : {demand_signal['mean_predicted_util'].mean():.4f}")
    surge_count = (demand_signal["pricing_recommendation"] == "surge").sum()
    discount_count = (demand_signal["pricing_recommendation"] == "discount").sum()
    standard_count = (demand_signal["pricing_recommendation"] == "standard").sum()
    print(f"Surge pricing zones: {surge_count} | discount: {discount_count} | standard: {standard_count}")
 
#running pricing agent

    print(f"\nTARIFF PRICING AGENT")
    print(f"Input: DemandSignal (predicted_util per gridID × hour)")
    print(f"  Output : bandit_tariff_kwh + rule_tariff_kwh per district-hour")
 
    tariff_decisions = pricing_agent.run_episode(demand_signal)
 
    print(f"Decisions : {len(tariff_decisions)} district-hour rows")
    print(f"Mean bandit tariff : ₹{tariff_decisions['bandit_tariff_kwh'].mean():.2f}/kWh")
    print(f"Mean rule tariff : ₹{tariff_decisions['rule_tariff_kwh'].mean():.2f}/kWh")
    print(f"Overall bandit revenue gain : {pricing_agent.short_term_memory['overall_revenue_gain']:+.2f}%")

    #running monitor agent
    print(f"\n MONITOR & LEARNING AGENT")

    metrics = monitor_agent.run_episode({
        "tariff_decisions": tariff_decisions,
        "acn_df":acn,
        "urban_df":episode_urban})
 
    all_metrics.append(metrics)
    print(f"\n Episode {episode} complete.")

#print the metrics
summary_cols = ["episode", "revenue_gain_pct", "charger_util_change","offpeak_uplift_pct", "wait_reduction_minutes","customer_response_rate", "pricing_efficiency_score"]

metrics_df = pd.DataFrame(all_metrics)
print_cols = [c for c in summary_cols if c in metrics_df.columns]
print(metrics_df[print_cols].to_string(index=False))
print(metrics_df.head())


