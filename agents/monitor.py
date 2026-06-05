

# making the monitor agent
# for this post-pricing utilization is simulated using price elasticity so that we can measure the actual pricing impact.This is standard econometric practice when you don't have a live A/B test

import json
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from agents.base_agent import BaseAgent

# defining constants

PRICE_ELASTICITY = -0.3
# I have chosen this value bcoz EV charging demand is relatively inelastic , users have limited alternatives and must charge. Literature estimates range from -0.2 to -0.4 so I use -0.3 as a conservative mid-estimate.

QUEUE_TO_MINUTES = 5
#Each unit of queue_length_proxy represents excess demand over capacity so assume that  1 excess session adds ~5 minutes of waiting time to subsequent users

ACN_BASELINE_REVENUE = 134.14
#Mean of (kWhDelivered × ₹15) across all 14,848 acn sessions.

ACN_AVG_KWH = 9.0
# Mean kWhDelivered from acn dataset

class MonitorAgent(BaseAgent):
 
    def __init__(self, pricing_agent):
        super().__init__("monitor_agent")
        self.pricing_agent = pricing_agent   # to get direct reference for q values feedback
        self.episode_history = self.recall("episode_history", [])

#make observe function of agent
    def observe(self, data: dict):   
        #the monitor agent will receive data from pricing agent that is tariff decisiions, acn data from acn preprocessed datset and urban ev data from urban preprocessed dataset 

        self.short_term_memory["tariff_decisions"] = data["tariff_decisions"]
        self.short_term_memory["acn_df"] = data["acn_df"]
        self.short_term_memory["urban_df"]= data["urban_df"]    
        return data
    
#now making the think function in which all the required metrics will be computed
    def think(self):  
        decisions = self.short_term_memory["tariff_decisions"].copy()
        acn = self.short_term_memory["acn_df"].copy()
        urban = self.short_term_memory["urban_df"].copy()
 
        metrics = {}  
 
 # revenue gain% metric
        total_sessions = decisions["total_volume"].sum()
        baseline_revenue = total_sessions * ACN_BASELINE_REVENUE

        dynamic_revenue  = (decisions["bandit_tariff_kwh"] * ACN_AVG_KWH * decisions["total_volume"]).sum()

        revenue_gain_pct = ((dynamic_revenue - baseline_revenue) / baseline_revenue * 100 if baseline_revenue > 0 else 0.0)

        metrics["revenue_gain_pct"] = round(revenue_gain_pct, 3)
        metrics["total_dynamic_revenue"] = round(dynamic_revenue,2)
        metrics["total_baseline_revenue"]= round(baseline_revenue,2)

# charger utilisation rate metric (diffference between before and after pricing)
           
        decisions["price_change_pct"] = ((decisions["bandit_tariff_kwh"] - decisions["base_tariff_kwh"])/ decisions["base_tariff_kwh"] * 100)     

# create simulated util rates after pricing
        
        decisions["simulated_util_after"] = (decisions["predicted_util"]* (1 + PRICE_ELASTICITY * decisions["price_change_pct"] / 100)).clip(0, 1)

        util_before = decisions["predicted_util"].mean()
        util_after= decisions["simulated_util_after"].mean()
        util_change = util_after - util_before

        metrics["charger_util_before"] = round(util_before, 4)
        metrics["charger_util_after"] = round(util_after,4)
        metrics["charger_util_change"] = round(util_change, 4)

# metric for off peak uplift
        
        # find rows where pricing zone is dicount
        discount_rows = decisions[decisions["pricing_action"] == "discount"].copy()
        
        if len(discount_rows) > 0:
            discount_rows["price_drop_pct"] = abs(discount_rows["price_change_pct"])
            sessions_before = discount_rows["total_volume"].sum()
            
            #since there is discount the number of sessions will increase
            # and the new number of sessions can be calculated using price elasticity and price drop
            discount_rows["sessions_after"] = (discount_rows["total_volume"]
                * (1 + (-PRICE_ELASTICITY) * discount_rows["price_drop_pct"] /100))
            
            sessions_after  = discount_rows["sessions_after"].sum()

            offpeak_uplift  = ((sessions_after - sessions_before) / sessions_before * 100 if sessions_before > 0 else 0.0)

        else:
            sessions_before = 0
            sessions_after  = 0
            offpeak_uplift  = 0.0  

        metrics["offpeak_sessions_before"] = round(sessions_before, 1)
        metrics["offpeak_sessions_after"] = round(sessions_after,1)
        metrics["offpeak_uplift_pct"] = round(offpeak_uplift,3)      

# now creating metric for avg time reduction

        if "queue_length_proxy" in urban.columns:
            queue_before = urban["queue_length_proxy"].mean()
        else:# calculate the queue legth proxy
            urban["queue_proxy_calc"] = (urban["demand_supply_ratio"] - 1 ).clip(lower=0)
            queue_before = urban["queue_proxy_calc"].mean() 

# now in surge zones after pricing changes util drops so queue drops so we can use util change ratio to find queue after pricing
        
        util_ratio = util_after / util_before if util_before > 0 else 1.0
        queue_after = queue_before * util_ratio
        queue_reduction  = queue_before - queue_after
        wait_reduction_hr= queue_reduction * QUEUE_TO_MINUTES / 60 #in hours

        metrics["queue_before"]  = round(queue_before,4)
        metrics["queue_after"] = round(queue_after,4)
        metrics["wait_reduction_minutes"] = round(wait_reduction_hr * 60, 2) 

        # metric for customer response rate
        avg_price_change = decisions["price_change_pct"].mean()
        elasticity_response = PRICE_ELASTICITY * avg_price_change
        customer_response_rate = round(elasticity_response, 3)
 
        metrics["avg_price_change_pct"] = round(avg_price_change, 3)
        metrics["customer_response_rate"] = customer_response_rate
        metrics["elasticity_assumption"] = PRICE_ELASTICITY

        #metric for pricing efficiency score

        dynamic_total_kwh =decisions["total_volume"] * ACN_AVG_KWH
        pricing_efficiency =dynamic_revenue/dynamic_total_kwh if dynamic_revenue>0 else 0

        baseline_efficiency = 15.0

        metrics["pricing_efficiency_score"] = round(pricing_efficiency, 4)
        metrics["baseline_efficiency"]  = baseline_efficiency
        metrics["efficiency_improvement"] = round(pricing_efficiency - baseline_efficiency, 4)

        self.short_term_memory["metrics"]   = metrics
        self.short_term_memory["decisions"] = decisions  # has simulated util columns so save in short term memory
        return metrics

# now define act function of agent
    def act(self):
        metrics= self.short_term_memory["metrics"]
        decisions = self.short_term_memory["decisions"]

        # what rewards to give to pricing agent based on revenue gain and congestion improvement

        # +1.0 : revenue went up and congestion went down
        # +0.5 : revenue went up
        # +0.2 : neutral or small improvement
        # -0.5 : revenue went down significantly
        # -1.0: revenue went down and congestion worsened

        rewards_sent = 0
        for _, row in decisions.iterrows():
            util_bin = row.get("util_bin", "medium")
            tariff = row.get("bandit_tariff_kwh", 15.0)
            rev_gain = row.get("revenue_gain_pct", 0)
            util_before = row.get("predicted_util", 0.5)
            util_after= row.get("simulated_util_after", util_before)
 
            congestion_improved = (util_before > 0.80 and util_after < util_before)


            if rev_gain > 5 and congestion_improved:
                reward = 1.0
            elif rev_gain > 0:
                reward = 0.5
            elif rev_gain > -5:
                reward = 0.2
            elif rev_gain < -10 and not congestion_improved:
                reward = -1.0
            else:
                reward = -0.5

            self.pricing_agent.update_q_value(util_bin, tariff, reward)
            rewards_sent += 1    

            print(type(self.pricing_agent))

# now add the info to episode history

        metrics["episode"] = self.episode_count
        metrics["timestamp"] = datetime.now().isoformat()
        self.episode_history.append(metrics)
        self.remember("episode_history", self.episode_history[-50:])

       
        # Export episode history

        os.makedirs("outputs", exist_ok=True)
        history_df = pd.DataFrame(self.episode_history)
        history_df.to_csv("outputs/metrics_summary.csv",index=False)

        self.log(
            action = "evaluate_episode",
            input_data= f"episode={metrics['episode']}, decisions={len(decisions)}",output_data= (f"rev_gain={metrics['revenue_gain_pct']:+.2f}%, "
                f"util_change={metrics['charger_util_change']:+.4f}, "
                f"offpeak_uplift={metrics['offpeak_uplift_pct']:+.2f}%"),reward = metrics["revenue_gain_pct"])
        
        return metrics
 


        

        
