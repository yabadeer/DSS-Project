import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# import data, clean and process
df = pd.read_csv("University_Data.csv")
df['Accept.Rate'] = df['Accept'] / df['Apps']
df["Total.Cost"] = df["Outstate"] + df["Room.Board"] + df["Books"] + df["Personal"]
df = df.rename(columns={'Unnamed: 0': 'Name'})
df['Private'] = df['Private'].map({'Yes': 1, 'No': 0})
columns_to_drop = ["Outstate", "Room.Board", "Books", "Personal", "Terminal", "Apps", "Accept", "Top25perc", "Enroll", "P.Undergrad", "Expend", "perc.alumni"]
df_clean = df.drop(columns=columns_to_drop)

# scaling feature columns
feature_cols = ['Private', 'Top10perc', 'F.Undergrad', 'PhD', 'S.F.Ratio', 'Grad.Rate', 'Accept.Rate', 'Total.Cost']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clean[feature_cols])

# training nearest neighbours
model = NearestNeighbors(n_neighbors=5)
model.fit(df_scaled)

# streamlit app
st.title('University Recommendation System')

# Collect user inputs
st.sidebar.header('Enter Your Preferences')

private = st.sidebar.selectbox('Private', [0, 1])
top10perc = st.sidebar.slider('Top 10% of H.S. Class', 0, 100, 50)
f_undergrad = st.sidebar.number_input('Number of Full-time Undergraduates', min_value=0, value=2000)
phd = st.sidebar.slider('Percentage of Faculty with PhD', 0, 100, 70)
s_f_ratio = st.sidebar.slider('Student-Faculty Ratio', 0.0, 50.0, 15.0)
grad_rate = st.sidebar.slider('Graduation Rate', 0, 100, 70)
accept_rate = st.sidebar.slider('Acceptance Rate', 0.0, 1.0, 0.8)
total_cost = st.sidebar.number_input('Total Cost', min_value=0, value=18000)

# Collect the inputs into a dictionary
student_preferences = {
    'Private': private,
    'Top10perc': top10perc,
    'F.Undergrad': f_undergrad,
    'PhD': phd,
    'S.F.Ratio': s_f_ratio,
    'Grad.Rate': grad_rate,
    'Accept.Rate': accept_rate,
    'Total.Cost': total_cost
}

# Convert user input to DataFrame
student_preferences_df = pd.DataFrame([student_preferences])

# Scale the user input using the same scaler
student_preferences_scaled = scaler.transform(student_preferences_df)

# Find the nearest neighbors
distances, indices = model.kneighbors(student_preferences_scaled)

# Get the recommended universities
recommended_universities = df_clean.iloc[indices[0]]

# Create a dictionary for renaming columns
rename_columns = {
    'Private': 'Private',
    'Top10perc': 'Top 10% of H.S. Class',
    'F.Undergrad': 'Number of Full-time Undergraduates',
    'PhD': 'Percentage of Faculty with PhD',
    'S.F.Ratio': 'Student-Faculty Ratio',
    'Grad.Rate': 'Graduation Rate',
    'Accept.Rate': 'Acceptance Rate',
    'Total.Cost': 'Total Cost'
}

# Rename the columns for display
recommended_universities_display = recommended_universities.rename(columns=rename_columns)

st.header('Recommended Universities:')
st.write(recommended_universities_display)
