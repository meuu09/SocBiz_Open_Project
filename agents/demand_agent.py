# making demand agent using the urban ev dataset

import pandas as pd
import numpy as np
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from agents.base_agent import BaseAgent


class DemandAgent(BaseAgent):
    def __init__(self):
        super().__init__("demand_agent")
        self.feature_columns = ["hour","day_of_week","is_weekend", "is_peak_hr","CBD", "dynamic_pricing","count", "fast_count","rolling_3h_volume", "occupancy_density", "queue_length_proxy"]
        self.target_column = "charger_util_rate"

        self.model      = None
        self.model_path = "memory/demand_model.pkl"
        self.load_model()

#load the model
    def load_model(self):
        # loads the model if it exists already
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
    
# save the loaded model    
    def _save_model(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)

#make the observe function of the agent
    def observe(self, data: pd.DataFrame):
        # will receive the UrbanEV processed dataframe
        self.short_term_memory["data"] = data

#make the think function in which we will make main model
    def think(self):
        data = self.short_term_memory["data"].copy()
        required = self.feature_columns + [self.target_column]
        data = data.dropna(subset=required)

#now split the data into train and test data
    
        X = data[self.feature_columns]
        y = data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

        self.model = XGBRegressor(n_estimators=300, learning_rate=0.05,max_depth=6,            subsample=0.8, colsample_bytree=0.8,  random_state=42, n_jobs=-1 )

        self.model.fit(X_train, y_train,eval_set=[(X_test, y_test)],verbose=False)

        pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test,pred)

        metrics = {"RMSE": round(rmse,4), "MAE": round(mae,4), "R2": round(r2,4)}
# add shap for explanation of role of features
        self.compute_shap(X_test, y_test)
# Save everything to memory
        self.remember("last_metrics", metrics)
        self._save_model()

#store the test results
        self.short_term_memory["X_test"] = X_test
        self.short_term_memory["y_test"] = y_test
        self.short_term_memory["metrics"] = metrics


#make function to find shap
    def compute_shap(self, X_test, y_test):
        try:
            import shap
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            os.makedirs("outputs", exist_ok=True)
            os.makedirs("plots",exist_ok=True)
#SHAP is slow on large sets so use a sample of 500 rows only
            sample_size = min(500, len(X_test))
            X_sample = X_test.sample(sample_size, random_state=42)

            explainer   = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)

            #mean absolute shap values
            shap_importance = pd.DataFrame({'feature': X_sample.columns.tolist(),'mean_abs_shap': np.abs(shap_values).mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
 
            shap_importance.to_csv("outputs/shap_importance.csv", index=False)

# make bar plot to compare importance of features
            plt.barh(shap_importance['feature'][::-1],shap_importance['mean_abs_shap'][::-1],edgecolor='white')   
            plt.title("Shap feature importance (Mean absolute impact on charger_util_rate prediction)")   
            plt.xlabel("mean SHAP values")
            plt.ylabel("features")
            plt.savefig("plots/shap_feature_importance.png", dpi=150)
            plt.close()

        except ImportError:
            print("[DemandAgent] SHAP is not installed. Run: pip install shap")

        except Exception as e:
            print(f"[DemandAgent] SHAP skipped: {e}")    


            


#make act function
    def act(self):
        data = self.short_term_memory["data"].copy()
        data = data.dropna(subset=self.feature_columns)  

        data["predicted_util"] = self.model.predict(data[self.feature_columns])
        data["predicted_util"] = data["predicted_util"].clip(0, 1)

        data["congestion_prob"] = (data   ["predicted_util"] > 0.80).astype(float)
        data["is_discount_zone"] = (data["predicted_util"] < 0.30).astype(float)

#make demand signal to be given to pricing agent
        demand_signal = data.groupby("gridID","hour").agg(
            mean_predicted_util = ("predicted_util","mean"),
            max_predicted_util = ("predicted_util", "max"),
            congestion_prob = ("congestion_prob","mean"),
            discount_prob = ("is_discount_zone","mean"),
            total_volume = ("volume","sum"),
            is_CBD = ("CBD","first"),has_dynamic_pricing= ("dynamic_pricing","first"),
            total_chargers = ("count","first")).reset_index()

# determine the price zone
        demand_signal["pricing_recommendation"]="standard"
        demand_signal.loc[demand_signal["congestion_prob"] > 0.3,"pricing_recommendation"] = "surge"
        demand_signal.loc[demand_signal["discount_prob"]> 0.6, "pricing_recommendation"] = "discount"
# Save predictions for output
        os.makedirs("outputs", exist_ok=True)
        data[["gridID","datetime","charger_util_rate","predicted_util","congestion_prob"]].to_csv("outputs/demand_predictions.csv", index=False)

        self.log("predict_demand",f"grids={len(demand_signal)}",f"mean_util={demand_signal["mean_predicted_util"].mean():.3f}",reward=None)
         
        self.short_term_memory['demand_signal'] = demand_signal

        return demand_signal