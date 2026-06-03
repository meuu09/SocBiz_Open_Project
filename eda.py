import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

acn_preprocessed=pd.read_csv("final_datasets\\final_acn_data.csv")
urbanev_preprocessed=pd.read_csv("final_datasets\\aligned_urbanev_data.csv")

#set colours for eda
colors={"surge":"#E63946",
        "standard":"#457B9D",
        "discount":"#2A9D8F"}


# EDA for acn data
#plot to see number of sessions by hour of day and see what are peak hours
sessions_by_hrs=acn_preprocessed.groupby("hour_of_day").size().reset_index(name="sessions")

print(sessions_by_hrs.head())

plt.figure(figsize=(12,5))
plt.bar(sessions_by_hrs["hour_of_day"], sessions_by_hrs["sessions"])
plt.xlabel("Hour of Day")
plt.ylabel("Number of Sessions")
plt.title("Charging Sessions by Hour")
plt.show()

#we observe that 14,15,16,17 have high demand and 4 to 13 have less demand 

#classifying hours 

bar_colors=[
    colors["surge"] if h in [14,15,16,17]
    else colors["discount"] if h in [4,5,6,7,8,9,10,11,12,13]
    else colors["standard"]
    for h in sessions_by_hrs["hour_of_day"]
]
surge_patch=patches.Patch(color="red",label="surge hours")
discount_patch=patches.Patch(color="green",label="discount hours")
standard_patch=patches.Patch(color="blue",label="standard hours")

plt.figure(figsize=(12,5))
plt.bar(sessions_by_hrs["hour_of_day"],sessions_by_hrs["sessions"],color=bar_colors,width=0.5,)
plt.xlabel("hour of day")
plt.ylabel("number of sessions")
plt.title("classifying hours by number of sessions")
plt.legend(handles=[surge_patch,discount_patch,standard_patch])
plt.xticks(range(24))
plt.show()


# plot to see number of sessions by days of week to see number of sessions in a particular day

sessions_by_day=acn_preprocessed.groupby("day_of_week").size().reset_index(name="sessions")
day_names=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
sessions_by_day["day_name"]=sessions_by_day["day_of_week"].map(dict(enumerate(day_names)))

plt.figure(figsize=(12,5))
plt.bar(sessions_by_day["day_name"], sessions_by_day["sessions"])
plt.xlabel("day name")
plt.ylabel("Number of Sessions")
plt.title("Charging Sessions by day")
plt.show()

# we see that weekdays have high demand and weekends have low demand
#classifying days
bar_colors=[colors["surge"] if d<5 
            else colors["discount"]
            for d in sessions_by_day["day_of_week"]]
weekday_patch=patches.Patch(color=colors["surge"],label="weekday (higher demand)")
weekend_patch=patches.Patch(color=colors["discount"],label="weekend (lower demand)")

plt.figure(figsize=(12,5))
plt.bar(sessions_by_day["day_name"],sessions_by_day["sessions"],color=bar_colors,width=0.5,)
plt.xlabel("day name")
plt.ylabel("number of sessions")
plt.title("classifying days by number of sessions")
plt.legend(handles=[weekday_patch,weekend_patch])
plt.xticks(sessions_by_day["day_name"])
plt.show()


# plot for showing kWh distribution
plt.figure(figsize=(10,5))
plt.hist(acn_preprocessed["kWhDelivered"].clip(0,60),bins=50,color=colors["standard"],edgecolor="white")

plt.axvline(acn_preprocessed["kWhDelivered"].mean(),color=colors["surge"],linewidth=2,label="mean kWh",linestyle="--")
plt.axvline(acn_preprocessed["kWhDelivered"].median(),color=colors["discount"],linewidth=2,label="median kWh",linestyle="--")
plt.title("distribution of energy delivered per session")
plt.xlabel("kwh delivered")
plt.ylabel("number of sessions")
plt.legend()
plt.show()


# stacked bar plot to see charging vs idle hours
print(acn_preprocessed["time_of_day"].isnull().sum())

time_of_day_stats=acn_preprocessed.groupby("time_of_day",observed=True).agg(avg_charging=("charging_duration_hr","mean"),avg_idle = ("idle_time_hr", "mean")).reset_index()

time_of_day_order = ["night","morning","afternoon","evening","late_night"]
time_of_day_stats["time_of_day"]=pd.Categorical(time_of_day_stats["time_of_day"],categories=time_of_day_order,ordered=True)
time_of_day_stats=time_of_day_stats.sort_values("time_of_day").reset_index().drop(columns=["index"])
print(time_of_day_stats.head())

plt.figure(figsize=(10,5))
x=range(len(time_of_day_stats))
plt.bar(time_of_day_stats["time_of_day"],time_of_day_stats["avg_charging"],label="active charging time",color="blue",alpha=0.5)
plt.bar(x,time_of_day_stats["avg_idle"],label="idle time",color="red",bottom=time_of_day_stats["avg_charging"],alpha=0.5)
plt.title("avg charging time vs avg idle time by time of day")
plt.xlabel("time of day")
plt.xticks(x)
plt.ylabel("hours")
plt.legend()

for i,row in time_of_day_stats.iterrows():
    total_charging_time=row["avg_charging"]+ row["avg_idle"]
    idle_percentage=row["avg_idle"]/ total_charging_time*100 if total_charging_time>0 else 0
    plt.text(x[i],total_charging_time+0.05,f"{idle_percentage:.2f}% idle",ha="center")

plt.show()

# plot to see idle time distribution along with its opportunity costs
fig, axe = plt.subplots(figsize=(10, 5))

axe.hist(
    acn_preprocessed["idle_time_hr"].clip(0, 12),
    bins=50,
    color=colors["surge"],
    alpha=0.85,edgecolor="white"
) 
# shows number of sessions a particular number of idle hours
axe.axvline(acn_preprocessed["idle_time_hr"].mean(), color="black", linestyle="--", linewidth=2,
                label=f"Mean idle hrs")
axe.set_title(f"Distribution of Idle Time per Session\n Total opportunity cost is {acn_preprocessed["idle_opportunity_cost"].sum():.2f} units")
axe.set_xlabel("Idle Hours")
axe.set_ylabel("Sessions")
axe.legend()
plt.show()

#revenue per session distribution at baseline price
plt.figure(figsize=(10,5))
plt.hist(acn_preprocessed["revenue_per_session"].clip(0,800),bins=50,color=colors["standard"],alpha=0.5,edgecolor="white")
plt.axvline(acn_preprocessed["revenue_per_session"].mean(),color="black",linestyle="--" ,linewidth=2,
           label="Mean revenue per session")
plt.axvline(acn_preprocessed["revenue_per_session"].median(),color="red",linestyle="--" ,linewidth=2,
           label="Median revenue per session")
plt.title("revenue per session distribution")
plt.xlabel("revenue")
plt.ylabel("sessions")
plt.legend()
plt.show()


# EDA for Urban ev data 

# demand heatmap plot for each hour of each day of week
pivot=urbanev_preprocessed.groupby(["hour","day_of_week"])["charger_util_rate"].mean().unstack()
print(pivot.head())
pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

plt.figure(figsize=(10,5))
sns.heatmap(pivot,cmap="YlOrRd",linewidths=0.3, linecolor="white",cbar_kws={"label":"mean charger util rate"},fmt=".2f", annot=True)
plt.title("Charger utilisation rate by hour and by day of week")
plt.xlabel("Day of week")
plt.ylabel("Hour of day")
plt.show()

#heatmap to see surge and discount windows

surge_pivot=urbanev_preprocessed.groupby(["hour","day_of_week"])["is_congested"].mean().unstack()
discount_pivot = urbanev_preprocessed.groupby(["hour","day_of_week"])["is_underutilized"].mean().unstack()
surge_pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
discount_pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
 
fig,axes = plt.subplots(1, 2, figsize=(12, 7)) 
sns.heatmap(surge_pivot,cmap="Reds",ax=axes[0],linewidths=0.3,cbar_kws={"label":"% of time congested (>80% util)"},fmt=".2f",annot=True)
axes[0].set_title("Surge pricing windows")
axes[0].set_xlabel("day")
axes[0].set_ylabel("hour")
sns.heatmap(discount_pivot,cmap="Blues",ax=axes[1],linewidths=0.3,cbar_kws={"label":"% of time underutilized(<30% util)"},fmt=".2f",annot=True)
axes[1].set_title("Discount pricing windows")
axes[1].set_xlabel("day")
axes[1].set_ylabel("hour")
plt.show()

# plot for showing charger utilization distribution

plt.figure(figsize=(10,5))
n,bins,bars=plt.hist(urbanev_preprocessed["charger_util_rate"].clip(0,1), bins=50,edgecolor="white")
#color the bars/patches by zones
for bar, left in zip(bars, bins[:-1]):
    if left < 0.30:
        bar.set_facecolor(colors["discount"])
    elif left > 0.80:
        bar.set_facecolor(colors["surge"])
    else:
        bar.set_facecolor(colors["standard"])

plt.axvline(0.30, color="black", linestyle="--", linewidth=2,label="Discount threshold 30%")     
plt.axvline(0.80, color=colors["surge"], linestyle="--", linewidth=2,label="Surge threshold 80%")     
plt.title("Charger utilisation distribution")
plt.xlabel("Charger Utilisation Rate")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# plot for seeing demand comparison in CBD (Central Business District) and suburbs
cbd_hourly_util=urbanev_preprocessed[urbanev_preprocessed["CBD"]==1].groupby("hour")["charger_util_rate"].mean().reset_index(name="util_rate")

suburb_hourly_util=urbanev_preprocessed[urbanev_preprocessed["CBD"]==0].groupby("hour")["charger_util_rate"].mean().reset_index(name="util_rate")
print(cbd_hourly_util.head())

plt.figure(figsize=(10,5))
plt.plot(cbd_hourly_util["hour"],cbd_hourly_util["util_rate"],color=colors["surge"],linewidth=2.5,marker="o",markersize=4,label="CBD (City Centre)")

plt.plot(suburb_hourly_util["hour"],suburb_hourly_util["util_rate"],color=colors["standard"],linewidth=2.5,marker="s",markersize=4,label="Suburban")

plt.axhline(0.80, color=colors["surge"],linestyle=":", alpha=0.5, label="Surge threshold (80%)")
plt.axhline(0.30, color=colors["discount"], linestyle=":", alpha=0.5, label="Discount threshold (30%)")

plt.title("CBD vs Suburban charger utilization rate")
plt.xlabel("hour of day")
plt.xticks(range(24))
plt.ylabel("mean utilisation rate")
plt.legend()
plt.show()

#plots to show difference between utilization rates and revenue between districts having synamic pricing and those having static pricing

dynamic_hourly_util=urbanev_preprocessed[urbanev_preprocessed["dynamic_pricing"]==1].groupby("hour")["charger_util_rate"].mean()
static_hourly_util=urbanev_preprocessed[urbanev_preprocessed["dynamic_pricing"]==0].groupby("hour")["charger_util_rate"].mean()

dynamic_revenue=urbanev_preprocessed[urbanev_preprocessed["dynamic_pricing"]==1].groupby("hour")["revenue_per_session"].mean()
static_revenue=urbanev_preprocessed[urbanev_preprocessed["dynamic_pricing"]==0].groupby("hour")["revenue_per_session"].mean()

#for utilisation rates
plt.figure(figsize=(10,5))
plt.plot(dynamic_hourly_util.index,dynamic_hourly_util.values,color=colors["surge"],linewidth=2.5, label="Dynamic pricing districts")
plt.plot(static_hourly_util.index,static_hourly_util.values,color=colors["standard"],linewidth=2.5, label="Static pricing districts")
plt.title("Charger Utilization Rate in Dynamic vs Static pricing districts")
plt.xlabel("hour")
plt.ylabel("mean utilization rate")
plt.legend()
plt.show()

#for revenue
plt.figure(figsize=(10,5))
plt.plot(dynamic_revenue.index,dynamic_revenue.values,color=colors["surge"],linewidth=2.5, label="Dynamic pricing districts")
plt.plot(static_revenue.index,static_revenue.values,color=colors["standard"],linewidth=2.5, label="Static pricing districts")
plt.title("Revenue in Dynamic vs Static pricing districts")
plt.xlabel("hour")
plt.ylabel("mean revenue")
plt.legend()
plt.show()

# per hour revenue comparison between districts with dynamic and the ones with static pricing
revenue_by_hour = urbanev_preprocessed.groupby("hour").agg(
    dynamic_rev = ("actual_total_revenue","sum"),
    base_rev = ("baseline_total_revenue","sum")
).reset_index()
print(revenue_by_hour.head())
print(urbanev_preprocessed["actual_total_revenue"].sum())
print(urbanev_preprocessed["baseline_total_revenue"].sum())
revenue_by_hour["revenue_gain_perct"] = (
    (revenue_by_hour["dynamic_rev"] - revenue_by_hour["base_rev"])
    / revenue_by_hour["base_rev"].replace(0, np.nan) * 100
).fillna(0)


#plot for absolute revenue
plt.figure(figsize=(10,5))
plt.plot(revenue_by_hour["hour"], revenue_by_hour["base_rev"] / 1e6, color=colors["standard"], linewidth=2.5, label="Static pricing")
plt.plot(revenue_by_hour["hour"], revenue_by_hour["dynamic_rev"] / 1e6, color=colors["surge"], linewidth=2.5,  label="Dynamic pricing")
plt.title("Revenue by hour Dynamic vs Static pricing districts")
plt.xlabel("hour")
plt.xticks(range(24))
plt.ylabel("revenue")
plt.legend()
plt.show()
print(urbanev_preprocessed["is_congested"].sum())
print(urbanev_preprocessed["is_underutilized"].sum())
# we see revenue for dynamic pricing is less than for static pricing because the number of underutilized districts is more than congested ones
#plot for %gain or loss per hour

colors_bar = [colors["surge"] if v >= 0 else colors["discount"] for v in revenue_by_hour["revenue_gain_perct"]]
plt.figure(figsize=(10,5))
plt.bar(revenue_by_hour["hour"],revenue_by_hour["revenue_gain_perct"],color=colors_bar,edgecolor="white")
plt.axhline(0,color="black", linewidth=1)
plt.title("Revenue %gain or loss by hour Dynamic vs Static pricing districts")
plt.xlabel("hour")
plt.xticks(range(24))
plt.ylabel("revenue gain%")
plt.legend()
plt.show()

#box plots for analysing demand volatality that is utilisationn variability
#for this the day is divided into three periods - peak , shoulder and off peak hours
#plot to see peak,shoulder and off peak hours
util_rate_by_hour = urbanev_preprocessed.groupby("hour")["charger_util_rate"].mean().reset_index()

plt.figure(figsize=(10,5))
plt.plot(util_rate_by_hour["hour"],util_rate_by_hour["charger_util_rate"],marker="o",linewidth=2
)
plt.title("Average Charger Utilization by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Mean Utilization Rate")
plt.xticks(range(24))
plt.legend()
plt.show()

# Average utilization for each hour
util_by_hour = (
    urbanev_preprocessed.groupby("hour")["charger_util_rate"].mean().reset_index(name="mean_util_rate"))

hourly_sorted = util_by_hour.sort_values("mean_util_rate",ascending=False
).reset_index(drop=True)
print(hourly_sorted)

#let us consider the hours with top 25% util rates as peak hours and those with lowermost 25% util rates as off peak hours
# Number of hours in 25%
n = int(len(hourly_sorted) * 0.25)

peak_hours = hourly_sorted.head(n)["hour"].tolist()
offpeak_hours = hourly_sorted.tail(n)["hour"].tolist()
shoulder_hours = [
    h for h in hourly_sorted["hour"]
    if h not in peak_hours + offpeak_hours
]

print("Peak hours:", peak_hours)
print("Off-peak hours:", offpeak_hours)
print("Shoulder hours:", shoulder_hours)


#so we can classify day by hours into peak:12am to 6am, shoulder:7am to 10 am and 7pm to 11pm,off peak as 10am to 6pm

#create function to make categories
def get_period(hour):
    if 0 <= hour < 7:
        return "Peak (12am–7am)"

    elif 7 <= hour < 10 or 19 <= hour <= 23:
        return "Shoulder (7–10am, 7–11pm)"

    else:
        return "Off-Peak (10am–6pm)"

urbanev_preprocessed["demand_period"]=urbanev_preprocessed["hour"].apply(get_period)

period_order=["Peak (12am–7am)","Shoulder (7–10am, 7–11pm)","Off-Peak (10am–6pm)"]

plt.figure(figsize=(10,5))
period_data = [urbanev_preprocessed[urbanev_preprocessed["demand_period"]==p]["charger_util_rate"].dropna().values for p in period_order]
boxplot=plt.boxplot(period_data,tick_labels=period_order,patch_artist=True)

boxcolors=[colors["surge"],colors["standard"],colors["discount"]]
for box,color in zip(boxplot["boxes"],boxcolors):
    box.set_facecolor(color)
    box.set_alpha(0.7)

plt.title("demand volatility by period")
plt.xlabel("demand period")
plt.ylabel("Charger Utilization Rate")
plt.legend()
plt.show()    

#plot to see queue length proxy by hours

queue_by_hour = urbanev_preprocessed.groupby('hour')['queue_length_proxy'].mean()
plt.figure(figsize=(10,5))
bar_colors = [colors['surge'] if v > queue_by_hour.mean() else colors['standard'] for v in queue_by_hour.values]
plt.bar(queue_by_hour.index, queue_by_hour.values, color=bar_colors, edgecolor='white')
plt.axhline(queue_by_hour.mean(), color='black', linestyle='--', linewidth=1.5,label='Mean queue proxy')
plt.title("queue length proxy by hour")
plt.xlabel("hour of day")
plt.xticks(range(24))
plt.ylabel("Queue length proxy")
plt.legend()
plt.show() 

#plot to see range of price multiplier

plt.figure(figsize=(10,5))
n, bins, patches = plt.hist(urbanev_preprocessed['price'], bins=60,edgecolor='white')

for patch, left in zip(patches, bins[:-1]):
    if left < 0.80:   
        patch.set_facecolor(colors['discount'])
    elif left > 1.10:
        patch.set_facecolor(colors['surge'])
    else:
        patch.set_facecolor(colors['standard'])

plt.axvline(1.0, color='black',linestyle='-',  linewidth=2,  label='Base (₹15.00/kWh multiplier=1.0)')
plt.axvline(0.25,color=colors['discount'],linestyle='--', linewidth=1.5,label='Min observed: ×0.25 = ₹3.75/kWh')
plt.axvline(1.47,color=colors['surge'], linestyle='--', linewidth=1.5,label='Max observed: ×1.47 = ₹22.05/kWh')
plt.title("Distribution of Price Multipliers")
plt.xlabel("Price Multiplier")
plt.ylabel("Frequency")
plt.legend()
plt.show() 