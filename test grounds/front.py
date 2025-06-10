import streamlit as st
import numpy as np
import plotly.express as px
import engine
import plotly.graph_objects as go
import altair as alt
import requests
import tempfile
import time
from apikey import data_location




from pyspark.sql import SparkSession

### ------------------ CACHED SETUP ------------------

@st.cache_resource
def get_spark_session():
    return SparkSession.builder.appName("Museh PySpark Learning").getOrCreate()

@st.cache_resource
def load_data():
    return spark.read.json(data_location)


@st.cache_resource
def get_clean_data():
    return engine.clean(df=load_data())

@st.cache_data
def get_artist_list(df, threshold=1000):
    return engine.get_artist_over_1000(df=df, number_of_lis=threshold)

@st.cache_data
def get_top_artists_by_state(_df, state):
    return engine.get_top_10_artists(df=_df, state=state)

@st.cache_resource
def get_map_data(_df, artist):
    return engine.get_artist_state(df=_df,artist=artist)

@st.cache_data
def kpis(_df):
    return engine.calculate_kpis(df=_df)

@st.cache_data
def user_list(_df, state):
    return engine.get_user_list(df = _df, state=state)

@st.cache_data
def top_paid(_df, state):
    return engine.top_paid_songs(df=_df, state=state)

@st.cache_data
def top_free(_df, state):
    return engine.top_free_songs(df=_df, state=state)

@st.cache_data
def create_pie(_df, state):
    return engine.create_subscription_pie_chart(df=_df, state=state)

@st.cache_data
def top_50_list(_df, state):
    return engine.get_top_50(df=_df, state=state)

@st.cache_data
def gen_ai_summary(artist_list):
    return engine.gen_genre_ai(artist_list=artist_list)

### ------------------ INITIAL STATE ------------------

if "option" not in st.session_state:
    st.session_state.option = "Kings Of Leon"

if "location" not in st.session_state:
    st.session_state.location = "Nationwide"

### ------------------ PAGE CONFIG ------------------
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center;'><span style='color: white'>Muse</span><span style='color: #87CEEB;'>Dash</span></h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Pipeline", "Dashboard", "Repo"])

spark = get_spark_session()
clean_listen = get_clean_data()

### ------------------ MAP RENDER FUNCTION ------------------

def render_map(artist):
    c = get_map_data(clean_listen, artist)
    fig = go.Figure(data=go.Choropleth(
        locations=c.state,
        z=c.listens,
        locationmode='USA-states',
        colorscale='Blues',
        colorbar_title="Number of\n Listens"
    ))
    fig.update_layout(geo_scope='usa', margin={"r": 0, "t": 0, "l": 0, "b": 0})
    event = st.plotly_chart(fig, on_select="rerun", selection_mode=["points", "box", "lasso"])
    points = event["selection"].get("points", [])
    if points:
        selected_state = points[0]["location"]
        if selected_state != st.session_state.location:
            st.session_state.location = selected_state
            st.rerun()
    else:
        if st.session_state.location != "Nationwide":
            st.session_state.location = "Nationwide"
            st.rerun()

### ------------------ MAIN UI: TAB 1 ------------------
with tab1:
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.image('MuseDash_Pipeline.png', caption='Pipeline')

### ------------------ MAIN UI: TAB 2 ------------------
with tab2:
    # First Row: Top 10 Artists and Map
    col_top = st.columns([6, 6], gap='large')
    with col_top[0]:
        with st.container(border=True):
            top_10 = get_top_artists_by_state(clean_listen, st.session_state.location)
            state_text = st.session_state.location if st.session_state.location != "Nationwide" else "the Nation"
            st.header(f"Top 10 Artists in {state_text}")
            selected_row = st.dataframe(
                top_10,
                use_container_width=True,
                selection_mode="single-row",
                on_select="rerun",
                hide_index=True
            )
            rows = selected_row['selection'].get("rows", [])
            if rows:
                selected_artist = top_10.Artist[rows[0]]
                if selected_artist != st.session_state.option:
                    st.session_state.option = selected_artist
                    st.rerun()
    with col_top[1]:
        with st.container(border=True):
            st.subheader(f"Number of {st.session_state.option} Listens")
            render_map(st.session_state.option)

    # Second Row: KPIs and AI Summary
    col_middle = st.columns([4, 4, 4], gap='large')
    kpi_data = kpis(_df=clean_listen)
    with col_middle[0]:
        with st.container(border=True):
            st.metric("Total Users", f'{round(kpi_data[0]/1000)}k+')
    with col_middle[1]:
        with st.container(border=True):
            st.metric("Avg Listening", f"{round(kpi_data[1]/60)} MIN")
    with col_middle[2]:
        with st.container(border=True):
            st.metric("Total Paid Listening", f"{round(kpi_data[2]/3600000)}k+ H")
    st.divider()
    with st.container(border=True):
        st.subheader(f'Most Popular Genre in {state_text}')
        top_50 = top_50_list(clean_listen, st.session_state.location)
        summary = gen_ai_summary(top_50)
        st.text_area("AI-Generated Genre Summary", value=summary)

    # Third Row: Charts
    col_bottom = st.columns(3, gap='large')
    with col_bottom[0]:
        with st.container(border=True):
            paid_songs_df = top_paid(_df=clean_listen, state=st.session_state.location)
            st.subheader(f'Top Paid Songs')
            chart = alt.Chart(paid_songs_df).mark_bar().encode(
                x='listens:Q',
                y=alt.Y('song:N', sort='-x'),
                tooltip=['song', 'listens']
            ).properties(height=350)
            st.altair_chart(chart, use_container_width=True)
    with col_bottom[1]:
        with st.container(border=True):
            free_songs_df = top_free(_df=clean_listen, state=st.session_state.location)
            st.subheader(f'Top Free Songs')
            chart = alt.Chart(free_songs_df).mark_bar().encode(
                x='listens:Q',
                y=alt.Y('song:N', sort='-x'),
                tooltip=['song', 'listens']
            ).properties(height=350)
            st.altair_chart(chart, use_container_width=True)
    with col_bottom[2]:
        with st.container(border=True):
            pie_df = create_pie(_df=clean_listen, state=st.session_state.location)
            st.subheader(f"Subscriptions Breakdown")
            total = pie_df["count"].sum()
            pie_df["percentage"] = (pie_df["count"] / total) * 100
            chart = alt.Chart(pie_df).mark_arc().encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(field="subscription", type="nominal"),
                tooltip=['subscription', 'count']
            ).properties(height=350)
            st.altair_chart(chart, use_container_width=True)

### ------------------ MAIN UI: TAB 3 ------------------
with tab3:
    col1, col2, col3 = st.columns([1, 5, 1])
    with col2:
        st.image('MuseDash_QR.png', caption='Repo')