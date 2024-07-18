import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

st.set_page_config(layout="wide")

# import data, clean and process
df = pd.read_csv("University_Data.csv")
df['Accept.Rate'] = df['Accept'] / df['Apps']
df["Total.Cost"] = df["Outstate"] + df["Room.Board"] + df["Books"] + df["Personal"]
df = df.rename(columns={'Unnamed: 0': 'Name'})
columns_to_drop = ["Outstate", "Room.Board", "Books", "Private", "Personal", "Terminal", "Apps", "Accept", "Top10perc", "Top25perc", "Enroll", "P.Undergrad", "Expend", "perc.alumni"]
df_clean = df.drop(columns=columns_to_drop)

# scaling feature columns
feature_cols = ['F.Undergrad', 'PhD', 'S.F.Ratio', 'Grad.Rate', 'Accept.Rate', 'Total.Cost']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clean[feature_cols])

# training nearest neighbours
model = NearestNeighbors(n_neighbors=5)
model.fit(df_scaled)

# streamlit app
st.title('University Recommendation System')

# Collect user inputs
st.sidebar.header('Enter Your Preferences')

f_undergrad = st.sidebar.number_input('Number of Full-time Undergraduates', min_value=0, value=2000)
phd = st.sidebar.slider('Percentage of Faculty with PhD', 0, 100, 70)
s_f_ratio = st.sidebar.slider('Student-Faculty Ratio', 0.0, 50.0, 15.0)
grad_rate = st.sidebar.slider('Graduation Rate (%)', 0, 100, 70)
accept_rate = st.sidebar.slider('Acceptance Rate (%)', 0, 100, 80)
total_cost = st.sidebar.number_input('Yearly Budget ($)', min_value=0, value=18000)


# Collect the inputs into a dictionary
student_preferences = {
    'F.Undergrad': f_undergrad,
    'PhD': phd,
    'S.F.Ratio': s_f_ratio,
    'Grad.Rate': grad_rate,
    'Accept.Rate': accept_rate / 100,
    'Total.Cost': total_cost
}

# Convert user input to DataFrame
student_preferences_df = pd.DataFrame([student_preferences])

# Scale the user input using the same scaler
student_preferences_scaled = scaler.transform(student_preferences_df)

# Compute distances to all data points
all_distances = np.linalg.norm(df_scaled - student_preferences_scaled, axis=1)

# Normalize and convert distances to scores
max_distance = np.max(all_distances)
scores = 100* (1 - (all_distances / max_distance))

# Add scores to the DataFrame
df_clean['Score'] = scores

# Sort the DataFrame by scores in descending order
recommended_universities = df_clean.sort_values(by='Score', ascending=False).reset_index(drop=True)

# Reorder columns to put Score after Name
cols = ['Name', 'Score'] + [col for col in recommended_universities.columns if col not in ['Name', 'Score']]
recommended_universities = recommended_universities[cols]

# Adjust the index to start from 1
recommended_universities.index = recommended_universities.index + 1

# Create a dictionary for renaming columns
rename_columns = {
    'F.Undergrad': 'Number of Full-time Undergraduates',
    'PhD': 'Percentage of Faculty with PhD',
    'S.F.Ratio': 'Student-Faculty Ratio',
    'Grad.Rate': 'Graduation Rate',
    'Accept.Rate': 'Acceptance Rate',
    'Total.Cost': 'Total Cost',
    'Score': 'Suitability Score'
}

# Rename the columns for display
recommended_universities["Accept.Rate"] = recommended_universities["Accept.Rate"] * 100
recommended_universities_display = recommended_universities.rename(columns=rename_columns)


st.header('Recommended Universities:')
st.dataframe(recommended_universities_display.head(10), width=1500)