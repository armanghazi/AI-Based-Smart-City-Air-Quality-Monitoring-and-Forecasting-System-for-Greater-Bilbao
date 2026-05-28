import streamlit as st
import pandas as pd
import geopandas as gpd
import folium

from streamlit_folium import st_folium

st.title(
    "🌍 Smart City Air Quality Dashboard"
)

# Load data
