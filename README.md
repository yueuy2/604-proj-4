# 604-proj-4

We focus on PJM ("Pennsylvania-New Jersey-Maryland") and its 29 zones.

## Data
PJM portal: https://dataminer2.pjm.com/list
Historical"metered" data: https://dataminer2.pjm.com/feed/hrl_load_metered

## Three tasks:
- Forecast hourly loads: one is supposed to make (29 regions) x (10 days) x (24 hours) = 6,960 predictions in total. The goal is to minimize mean squared error.
- Predict which hour of the day will have the peak load (the "peak hour"): for each day and each region, dene the "peak hour" as the hour with the maximum load (for that day
and region). If your prediction is correct, +/- one hour; simple approach not necessarily optimal.
- Predict which days will have maximum peak loads (the "peak days")


https://www.kaggle.com/code/robikscube/starter-hourly-energy-consumption/comments
