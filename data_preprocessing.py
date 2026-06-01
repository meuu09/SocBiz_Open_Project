import numpy as np
import pandas as pd

acn_data=pd.read_excel("datasets\\acndata_sessions.json.xlsx")
adj_data=pd.read_csv("datasets\\adj.csv")
distance_data=pd.read_csv("datasets\\distance.csv")
duration_data=pd.read_csv("datasets\\duration.csv")
info_data=pd.read_csv("datasets\\information.csv")
occupancy_data=pd.read_csv("datasets\\occupancy.csv")
price_data=pd.read_csv("datasets\\price.csv")
stations_data=pd.read_csv("datasets\\stations.csv")
time_data=pd.read_csv("datasets\\time.csv")
volume_data=pd.read_csv("datasets\\volume.csv")

# check properties of the acn dataset
print(acn_data.shape)
print(acn_data.info())
print(acn_data.describe())
print(acn_data["paymentRequired"].value_counts())
print(acn_data["timezone"].value_counts())
print(acn_data["clusterID"].value_counts())

# drop the columns that are completely empty or not needed in analysis
acn_data.drop(columns=["_meta","end","min_kWh","site","start","_items","_id","userInputs","paymentRequired","timezone","modifiedAt","clusterID"],inplace=True)
print(acn_data.shape)
print(acn_data.head())

#drop the duplicate userid column
acn_data.drop(columns=["userID.1"],inplace=True)
print(acn_data.shape)

#drop the rows where we main variable values that is connectiontime,disconnecttime,etc are missing
acn_data.dropna(subset=["connectionTime","disconnectTime","kWhDelivered"],inplace=True)
print(acn_data.shape)

#parse the timestamps to datetime format
time_format = "%a, %d %b %Y %H:%M:%S GMT"
acn_data['connectionTime'] = pd.to_datetime(acn_data['connectionTime'], format=time_format, errors='coerce')
acn_data['disconnectTime']  = pd.to_datetime(acn_data['disconnectTime'], format=time_format, errors='coerce')
acn_data['doneChargingTime']= pd.to_datetime(acn_data['doneChargingTime'],format=time_format, errors='coerce')
print(acn_data.info())

print(acn_data[["connectionTime","disconnectTime","doneChargingTime"]].head())
# Drop rows where timestamp parsing failed
print(acn_data.shape)
acn_data.dropna(subset=['connectionTime', 'disconnectTime'], inplace=True)
print(acn_data.shape)
#kWhDelivered stats
print(acn_data["kWhDelivered"].describe())
#mean         9.002466
# min          0.501000
# max         69.373000

#calculating total time duaration for each session
acn_data["session_duration_hr"]=(acn_data["disconnectTime"]-acn_data["connectionTime"]).dt.total_seconds()/3600

#calculating the actual charging duration for eac session
acn_data["charging_duration_hr"]=(acn_data["doneChargingTime"]-acn_data["connectionTime"]).dt.total_seconds()/3600


#calculating the idle time for each session
acn_data["idle_time_hr"]=(acn_data["session_duration_hr"]-acn_data["charging_duration_hr"])
print(acn_data[["session_duration_hr","charging_duration_hr","idle_time_hr"]].info())
print(acn_data[["session_duration_hr","charging_duration_hr","idle_time_hr"]].describe())

# removing outliers

# keep only values where idle time duration is non negative 
print(acn_data.shape)
acn_data=acn_data[(acn_data["idle_time_hr"]>=0)]     
print(acn_data.shape)

# keep only values where charging time duration is non negative 

acn_data=acn_data[(acn_data["charging_duration_hr"]>=0)]     
print(acn_data.shape)


# keep only values where session duration is positive 

acn_data=acn_data[(acn_data["session_duration_hr"]>0)]     
print(acn_data.shape)
print(acn_data.head())

#remove rows where duration exceeds hours as they are outliers
acn_data=acn_data[(acn_data["session_duration_hr"]<=24)]
print(acn_data.shape) 

# remove rows where kWh delivered is greater than 100 as they are outliers . A standard EV battery is 40-100 kWh. Delivering >100 kWh in one session
# is physically impossible for a single car

acn_data=acn_data[(acn_data["kWhDelivered"]<=100)]
print(acn_data.shape)

#creating meaningful features

#charger utilisation rate

acn_data["charger_util_rate"]=acn_data["charging_duration_hr"]/acn_data["session_duration_hr"]
print(acn_data[["charger_util_rate"]].describe())

#revenue per session

#as given base tariff = 15.0
base_tariff=15.0
acn_data["revenue_per_session"]=base_tariff*acn_data["kWhDelivered"]

#energy cost per kWh( effective rate per kWh)
acn_data["energy_cost_per_kWh"]=acn_data["revenue_per_session"]/acn_data["kWhDelivered"]
#energy cost per kwh is actually same as base tariff for static model . It will change in dynamic pricing model
acn_data["energy_cost_per_kWh"]=base_tariff

# queue length proxy

#if at same time there are more than 1 sessions at the same station then there is a queue as demand is high

#sort the session timinings
acn_data=acn_data.sort_values("connectionTime").reset_index(drop=True)

# round the session time down to hours
acn_data["hour_slot"]=acn_data["connectionTime"].dt.floor("H")

#make a new dataframe for each station id and hour slot and count the number of sessions in each hour slot for each station id
queue_length=acn_data.groupby(["stationID","hour_slot"]).size().reset_index(name="queue_length_proxy")

#merge with acn_data
acn_data=acn_data.merge(queue_length,on=["stationID","hour_slot"],how="left")

# make a flag to indicate congestion
acn_data["isCongested"]=(acn_data["queue_length_proxy"]>1).astype(int)
print(acn_data["isCongested"].value_counts())
print(acn_data.head())

# occupancy density = total sessions at a station on a date / total hours in a day (24)

#a new column for date
acn_data["date"]=acn_data["connectionTime"].dt.date
# create a dataframe for total sessions on a particular date at each station
daily_sessions=acn_data.groupby(["stationID","date"]).size().reset_index(name="daily_sessions")
# merge with acn_data
acn_data=acn_data.merge(daily_sessions,on=["stationID","date"],how="left")
acn_data["occupancy_density"]=acn_data["daily_sessions"]/24

# idle opportunity cost ---revenue that could have been earned if there was no idle time
mean_kWh_delivered=9.0
mean_session_duration_hr=5.92
avg_kwh_per_hr=mean_kWh_delivered/mean_session_duration_hr

acn_data["idle_opportunity_cost"]=acn_data["idle_time_hr"]*avg_kwh_per_hr*base_tariff

# create time features to see when the demand is high or low
acn_data["hour_of_day"]=acn_data["connectionTime"].dt.hour
acn_data["date"]=acn_data["connectionTime"].dt.date
acn_data["day_of_week"]=acn_data["connectionTime"].dt.dayofweek
acn_data["month"]=acn_data["connectionTime"].dt.month
acn_data["is_weekend"]=(acn_data["day_of_week"]>=5).astype(int)

#from data we observed that demand is higher after noon
acn_data["is_peak_hour"]=((acn_data["hour_of_day"].isin([13,14,15,16,17]))).astype(int)

# group hours into 5 categories night (0-5), morning (6-11), afternoon (12-16), evening (17-21), late night (22-23)
acn_data["time_of_day"]=pd.cut(acn_data["hour_of_day"], bins=[-1,5,11,16,21,23], labels=["night","morning","afternoon","evening","late_night"])

#we have user info only in some rows that can be used in analysis. so create a flag to indicate if user info is available
acn_data["user_info_available"]=(acn_data["userID"].notna()).astype(int)

#also only some rows have user inputs like kWh requested, etc so create a flag for that as well
acn_data["user_inputs_available"]=(acn_data["kWhRequested"].notna()).astype(int)

print(acn_data.info())

acn_data=acn_data[[ "sessionID","stationID","siteID","spaceID","connectionTime","disconnectTime","doneChargingTime","kWhDelivered","session_duration_hr","charging_duration_hr","idle_time_hr","charger_util_rate","revenue_per_session","energy_cost_per_kWh","hour_slot","queue_length_proxy","isCongested","daily_sessions","occupancy_density","idle_opportunity_cost","date","hour_of_day","day_of_week","month","is_weekend","is_peak_hour","time_of_day","user_info_available","userID","user_inputs_available","WhPerMile","kWhRequested","milesRequested","minutesAvailable","requestedDeparture"]]
print(acn_data.isnull().sum())
print(acn_data.head(1))

