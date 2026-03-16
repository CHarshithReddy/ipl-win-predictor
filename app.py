import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="IPL Win Predictor", layout="wide")

@st.cache_resource
def load_models():
    with open('src/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('src/features.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/model_data.csv')

model, FEATURES = load_models()
df = load_data()

st.title("IPL Win Probability Predictor")
st.markdown("Ball by ball win prediction using XGBoost + SHAP")

# Sidebar
st.sidebar.title("Select a Match")
match_ids = df['match_id'].unique()
selected_match = st.sidebar.selectbox("Match ID", match_ids)

# Filter match
match_df = df[df['match_id'] == selected_match].copy()
match_df = match_df.sort_values('balls_remaining', ascending=False).reset_index(drop=True)

# Predict
probs = model.predict_proba(match_df[FEATURES])[:,1]
match_df['win_prob'] = probs
match_df['ball_number'] = 120 - match_df['balls_remaining']

# Metrics
current = match_df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Win probability", f"{current['win_prob']*100:.1f}%")
c2.metric("Runs required", f"{int(current['runs_required'])}")
c3.metric("Balls remaining", f"{int(current['balls_remaining'])}")
c4.metric("Wickets in hand", f"{int(current['wickets_in_hand'])}")

# Win probability chart
st.subheader("Win probability over time")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=match_df['ball_number'],
    y=match_df['win_prob'] * 100,
    mode='lines',
    name='Win probability',
    line=dict(color='#185FA5', width=2.5),
    fill='tozeroy',
    fillcolor='rgba(24,95,165,0.08)'
))
fig.update_layout(
    xaxis_title='Ball number',
    yaxis_title='Win probability (%)',
    yaxis=dict(range=[0,100]),
    height=400
)
st.plotly_chart(fig, use_container_width=True)

# SHAP explanation
st.subheader("Why this prediction?")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(current[FEATURES].to_frame().T)
shap_df = pd.DataFrame({
    'Feature': FEATURES,
    'Impact': shap_values[0]
}).sort_values('Impact')

import plotly.express as px
fig2 = px.bar(
    shap_df, x='Impact', y='Feature',
    orientation='h',
    color='Impact',
    color_continuous_scale=['#A32D2D', '#185FA5'],
    title='Feature impact on current prediction'
)
st.plotly_chart(fig2, use_container_width=True)