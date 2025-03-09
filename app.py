import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Fetch COVID-19 data for the USA
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extract relevant data
covid_data = {
    "Total Cases": data["cases"],
    "Today Cases": data["todayCases"],
    "Total Deaths": data["deaths"],
    "Today Deaths": data["todayDeaths"],
    "Recovered": data["recovered"],
    "Active Cases": data["active"],
    "Critical Cases": data["critical"],
    "Cases Per Million": data["casesPerOneMillion"],
    "Deaths Per Million": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])

# Streamlit App
st.title("COVID-19 Cases Prediction in USA")
st.subheader("Latest COVID-19 Statistics")
st.dataframe(df)  # Display the data table

# Bar Chart Visualization
st.subheader("COVID-19 Data Visualization")
fig, ax = plt.subplots(figsize=(8, 5))
labels = ["Total Cases", "Active Cases", "Recovered", "Total Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]
ax.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")
st.pyplot(fig)

# Generate Random Historical Data (Simulated Last 30 Days)
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)
historical_deaths = np.random.randint(500, 2000, size=30)
df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Train Linear Regression Model
X = df_historical[["day"]]
y = df_historical["cases"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# User Input for Prediction
st.subheader("Predict COVID-19 Cases for Future Days")

with st.form("prediction_form"):
    day_input = st.number_input("Enter day number (e.g., 31 for next prediction)", min_value=1, max_value=100, step=1)
    submit_button = st.form_submit_button("Predict")

    if submit_button:
        prediction = model.predict(np.array([[day_input]]))
        st.success(f"Predicted cases for day {day_input}: {int(prediction[0])}")
