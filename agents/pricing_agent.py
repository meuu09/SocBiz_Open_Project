# make the pricing agent that will predict the tariffs to be imposed in particular grids in particular hours using the demand signals from the demand agent and the revenue prediction from the revenue model from acn features

import pandas as pd
import numpy as np
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from agents.base_agent import BaseAgent

class PricingAgent(BaseAgent):
    TARIFF_OPTIONS = [8,10,12,15,18,21,24]  
    SURGE_THRESHOLD = 0.80
    DISCOUNT_THRESHOLD = 0.30
    #mean baseline revenue and kwh delivered calculated form acn datset
    ACN_BASELINE_REVENUE = 134.14 
    ACN_AVG_KWH = 9.0

    def __init__(self):
        super().__init__("pricing_agent")
        self.epsilon = 0.15  
        # means the model will 15% explore the unused tariff option and 85% exploit the tariff values with highest qvalues

        #revenue model to predict revenue
        self.revenue_model = None
        self.revenue_by_hour = {}  
        self.init_q_values()

#we will make categories for different range of util rates and then impose tariffs on those categories and decide the pricing zone depending on the q values 

#the categories will be as follows:

# very low util < 0.20 means very empty so big discount needed
# low util i.e. 0.20-0.40 means somewhat empty so mild discount
# medium util  i.e. 0.40-0.60 means normal so standard rate
# high util i.e. 0.60-0.80 means getting full → mild surge
# very_high utile i.e. > 0.80 means congested so big surge

    #make function to initialise q values
    def init_q_values(self):
        existing = self.recall('q_values')
        if existing:
            self.q_values= existing
            self.action_counts = self.recall('action_counts', {})
        # q values is score given to each tariff value for each category depending on its correctness
        #action count is number of times that tariff option is used

        else: #if q values not already existing set suitable q values and action counts for all categories depending upon the different zones
            self.q_values = {'very_low': {'8': 0.3, '10': 0.4, '12': 0.5, '15': 0.3, '18': 0.1, '21': 0.0, '24': 0.0},
            'low': {'8': 0.2, '10': 0.3, '12': 0.4, '15': 0.4, '18': 0.2, '21': 0.1, '24': 0.0},
            'medium': {'8': 0.0, '10': 0.1, '12': 0.2, '15': 0.5, '18': 0.2, '21': 0.1, '24': 0.0},
            'high':{'8': 0.0, '10': 0.0, '12': 0.1, '15': 0.2, '18': 0.5, '21': 0.4, '24': 0.3},
            'very_high': {'8': 0.0, '10': 0.0, '12': 0.0, '15': 0.1, '18': 0.3, '21': 0.5, '24': 0.6},}
            self.action_counts = {b: {str(t): 0 for t in self.TARIFF_OPTIONS} for b in self.q_values}

#make function to define categories or bins
    def util_to_bin(self, util):
        if util < 0.20:
            return 'very_low'
        elif util < 0.40:
            return 'low'
        elif util < 0.60:
            return 'medium'
        elif util < 0.80:
            return 'high'
        else:             
            return 'very_high'
        
#make function to select tariff value 
    def select_tariff(self, util_bin):
    
        if util_bin in ('very_high','high'):
            valid_tariffs = [15, 18, 21, 24]
        elif util_bin in ('very_low', 'low'):
            valid_tariffs = [8, 10, 12, 15]    # discount zone → only standard or below
        else:
            valid_tariffs = self.TARIFF_OPTIONS
        if np.random.rand() < self.epsilon:
            return float(np.random.choice(valid_tariffs))
        q = self.q_values[util_bin]
        bestq = max(valid_tariffs, key=lambda t: q[str(t)])
        return float(bestq)

#function to update q value
    def update_q_value(self,util_bin,tariff, reward, alpha=0.1):
       
    #    This function is by Monitor Agent with reward from actual outcomes and it updates q value: newq= oldq + alpha × (reward - oldq)
        
        t = str(int(tariff))
        if util_bin in self.q_values and t in self.q_values[util_bin]:
            self.action_counts[util_bin][t] += 1
            old_q = self.q_values[util_bin][t]
            self.q_values[util_bin][t] = old_q + alpha * (reward - old_q)
            
           
# make the revenue prediction model from acn dataset
    def train_revenue_model(self, acn_df: pd.DataFrame):
        feature_cols = ['hour','session_duration_hr', 'charging_duration_hr','is_peak_hour', 'is_weekend', 'charger_util_rate','queue_length_proxy', 'idle_time_hr', 'day_of_week']    

        target_col = 'revenue_per_session'

        available = [c for c in feature_cols if c in acn_df.columns]     
        data = acn_df.dropna(subset=available + [target_col]).copy()

        X = data[available]
        y = data[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.revenue_model = XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=5,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1)

        self.revenue_model.fit(X_train, y_train,eval_set=[(X_test, y_test)],verbose=False)

        pred = self.revenue_model.predict(X_test)
        rmse= np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        self.remember('revenue_model_metrics',{'RMSE': round(rmse, 2), 'R2': round(r2, 4)})
#make revenue by hour dataframe
        data['predicted_revenue'] = self.revenue_model.predict(X)
        self.revenue_by_hour = (data.groupby('hour')['predicted_revenue'].mean().to_dict())
        self.remember('revenue_by_hour', self.revenue_by_hour)

 # Save model
        model_path = "memory/pricing_revenue_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.revenue_model, f)
 
        return self.revenue_model        

#make observe function of the model- it observes the demand signal from demand agent
    def observe(self, demand_signal: pd.DataFrame):
      
        self.short_term_memory['demand_signal'] = demand_signal


#make think function of agent- takes the demand signal and the predicted reveneue and decides the bandit as well as rule tariff values
    def think(self):
        demand = self.short_term_memory['demand_signal'].copy()
        decisions = []

        for _, row in demand.iterrows():
            util = row['mean_predicted_util']
            util_bin = self.util_to_bin(util)
            hour = int(row['hour']) 
  
            predicted_rev_this_hour = self.revenue_by_hour.get(str(hour), self.ACN_BASELINE_REVENUE)

            bandit_tariff = self.select_tariff(util_bin)

            if util > self.SURGE_THRESHOLD:
                if predicted_rev_this_hour > self.ACN_BASELINE_REVENUE:
                    rule_tariff = 21.0   # high demand + revenue signal means full surge
                else:
                    rule_tariff = 18.0   # high demand but low revenue means cautious surge
                action = 'surge'

            
            elif util < self.DISCOUNT_THRESHOLD:
                if predicted_rev_this_hour < self.ACN_BASELINE_REVENUE * 0.8:
                    rule_tariff = 8.0    # very low demand + low rev means deep discount
                else:
                    rule_tariff = 12.0   # low demand but ok revenue means mild discount
                action = 'discount'  

            else:
                rule_tariff = 15.0
                action  = 'standard'     


            decisions.append({
                'gridID':row['gridID'],
                "hour":int(row["hour"]),
                'predicted_util': round(util, 4),
                'util_bin': util_bin,
                'pricing_action': action,
                'bandit_tariff_kwh': round(bandit_tariff, 2),
                'rule_tariff_kwh': round(rule_tariff, 2),
                'base_tariff_kwh': 15.0,
                'predicted_revenue_this_hour': round(predicted_rev_this_hour, 2),
                'total_volume':row['total_volume'],
                'is_CBD': row['is_CBD'],
                'total_chargers':row['total_chargers'],
                'congestion_prob': round(row['congestion_prob'], 4),
                'discount_prob': round(row['discount_prob'], 4),})   

        self.short_term_memory['decisions'] = pd.DataFrame(decisions)  

#now make the act function
    def act(self):
        d = self.short_term_memory['decisions'].copy()

#Revenue with bandit tariff
        d['bandit_rev_per_session'] = d['bandit_tariff_kwh'] * self.ACN_AVG_KWH
        d['bandit_total_revenue'] = d['bandit_rev_per_session'] * d['total_volume']

#Revenue with rule-based tariff
        d['rule_rev_per_session'] = d['rule_tariff_kwh'] * self.ACN_AVG_KWH
        d['rule_total_revenue'] = d['rule_rev_per_session'] * d['total_volume']

#Baseline revenue — ACN flat ₹15/kWh
        d['base_total_revenue'] = self.ACN_BASELINE_REVENUE * d['total_volume']


#calculate Revenue Gain %
        for col_new, col_base, col_gain in [
        ('bandit_total_revenue','base_total_revenue', 'bandit_revenue_gain_pct'),
        ('rule_total_revenue','base_total_revenue', 'rule_revenue_gain_pct')]:
            d[col_gain] = ((d[col_new] - d[col_base])/ d[col_base].replace(0, np.nan) * 100
            ).fillna(0).round(3) 

# Off-Peak sessions (for Uplift metric)
        d['off_peak_sessions']= d['total_volume'] * d['discount_prob']
        d['in_discount_zone']  = (d['predicted_util'] < self.DISCOUNT_THRESHOLD).astype(int)

        total_bandit = d['bandit_total_revenue'].sum()
        total_base = d['base_total_revenue'].sum()
        overall_gain = ((total_bandit - total_base) /total_base * 100
            if total_base > 0 else 0)
        
# Save the outputs
        os.makedirs("outputs", exist_ok=True)
        d.to_csv("outputs/tariff_recommendations.csv", index=False)

        self.log('set_tariffs',
                 f"grids={len(d)}",
                 f"revenue_gain={overall_gain:.2f}%",
                 reward=overall_gain)   

        self.short_term_memory['tariff_decisions'] = d
        self.short_term_memory['overall_revenue_gain'] = overall_gain
        self.short_term_memory['total_bandit_revenue']= total_bandit
        self.short_term_memory['total_base_revenue']   = total_base

        return d     