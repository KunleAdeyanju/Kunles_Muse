spark = SparkSession.builder \
    .appName("MuseDash PySpark") \
        .getOrCreate()

try:
    df_listen = spark.read.json ('/Users/kunle/Python Projects/Kunles_Muse/Data/listen_events')
    print('Data loaded successfully')
except Exception as e:
    print(f'Error loading data: {e}')

import streamlit as st
import numpy as np
import plotly.express as px
import engine
import plotly.graph_objects as go
import altair as alt
import requests
import tempfile
import time


from pyspark.sql import SparkSession

# makes page wide
st.set_page_config(layout = 'wide')

spark = SparkSession.builder \
    .appName("MuseDash PySpark") \
        .getOrCreate()

@st.cache_resource
def load_data():
    url = "https://zcw-cohort-spring25.s3.us-east-2.amazonaws.com/listen_events"

    try:
        # Download the file into a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            response = requests.get(url)
            response.raise_for_status()
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        # Start Spark session
        spark = SparkSession.builder.getOrCreate()

        # Load the JSON data into a Spark DataFrame
        df_listen = spark.read.json(tmp_file_path)
        print("Data loaded successfully from URL")

    except Exception as e:
        print(f"Error loading data: {e}")
        df_listen = None

    return df_listen

df_listen = load_data()

@st.cache_resource
def cleaned():
    cleaned_listen = engine.clean(df=df_listen)
    return cleaned_listen

cleaned_listen = cleaned()

# formatting transforming
cleaned_listen = engine.clean(df=df_listen)
artist_list = engine.get_artist_over(df=cleaned_listen,number_of_lis=1000)


                   
# Streamlit Titling
st.markdown("<h1 style='text-align: center;'><span style='color: white;'>Muse</span><span style='color: blue;'>Dash</span></h1>", unsafe_allow_html=True)



col_table = st.columns((5, 10), gap='medium')                                  


with st.container(border=True):


    with col_table[1]:
        with st.container(border=True):

            option = st.selectbox(
            'Filter by Artist',
            artist_list,
            #index=None,
            placeholder="Chosen Artist",
            accept_new_options = True
        )


            if option == None:
                with st.container():
                    st.write("You selected: ", option)
            else:
                # creating the dataframe of listens for specific artists
                b = engine.get_artist_state_listen(df=cleaned_listen, artist=option)

                # filtering data to what is needed to make map
                c = engine.map_prep_df(df=b)
                c_max = c['listens'].max()
                c_min = c['listens'].min()
                ## creating the maps
                fig = go.Figure(data=go.Choropleth(
                    locations=c.state, # Spatial coordinates
                    z = c.listens, # Data to be color-coded
                    locationmode = 'USA-states', # set of locations match entries in `locations`
                    colorscale = 'Blues',
                    #color = [c_min, c_max ],
                    colorbar_title = "Number of\n Listens",
                    
                ))

                # adding context to the map
                fig.update_layout(
                    title_text = f'Number of {option} Listens \n 2024-2025',
                    geo_scope='usa', # limit map scope to USA
                    margin={"r":0,"t":0,"l":0,"b":0} # setting margins around to 0s, filling the container as much as possible
                )

                event = st.plotly_chart(fig, on_select="rerun", selection_mode=["points","box","lasso"])

                # selects the dictionary where the state info is stored
                points = event["selection"].get("points", [])
                
                # sets the location based on which state is clicked
                if points:
                    first_point = points[0]
                    location = first_point['location']
                else:
                    location = 'Nationwide'
                
    


              
    
    selected_state = location
    # titles depending on state selected
    if selected_state == 'Nationwide':
        top_10_header = "Top 10 National Artists"
        pie_title = "National" + "\n" + "Subscriptions"
        paid_title = 'Top Songs for Paid Users'
        free_title = 'Top Songs for Free Users'
        chart_title = "How long are users listening in the USA?"
    else:
        top_10_header = f"Top 10 Artists in {selected_state}"
        pie_title = "Subscriptions" + "\n" + f"in {selected_state}"
        paid_title = f'Top Songs for Paid Users in {selected_state}'
        free_title = f'Top Songs for Free Users in {selected_state}'
        chart_title = f"How long are users listening in {selected_state}?"
        
        
    with col_table[0]:
        with st.container(border=True):
            # Printing top ten chart
            top_10 = engine.get_top_10_artists(df=cleaned_listen, state=selected_state)
            st.header(top_10_header)

            # Style the table
            styled_top_10 = top_10.style.hide(axis="index").set_table_styles([
                {'selector': 'td', 'props': [('font-size', '20px'), ('text-align', 'center')]},
                {'selector': 'th', 'props': [('font-size', '20px'), ('text-align', 'center')]}
            ])
            st.markdown(styled_top_10.to_html(), unsafe_allow_html=True)

        with st.container():
            # KPI metrics
            total_users, average_listening_time, total_duration_sum = engine.calculate_kpis(df=df_listen)
            col1, col2, col3 = st.columns([1.5, 2, 2.2])
            with col1:
                with st.container(border=True):
                    st.metric("Total Users", "1k+")
            with col2:
                with st.container(border=True):
                    st.metric("Average Total Listening", "4 MIN")
            with col3:
                with st.container(border=True):
                    st.metric("Total Paid Listening", "70k+ HR")

        with st.container(border=True):
                # listen graph creation
                listen_duration = engine.get_user_list(df=cleaned_listen, state=selected_state)
                    
                #create the line graph
                line_fig = px.line(
                    listen_duration,
                    x="month_name",
                    y="total_duration",
                    color="subscription",
                    title=chart_title,
                    labels={"month_name": "Month", "total_duration": "Total Duration (seconds)"}
                        )
                line_fig.update_layout(
                    hovermode="x unified",

                    #style the hover line color
                    xaxis=dict(
                        showspikes=True,
                        spikemode='across',
                        spikesnap='cursor',
                        spikethickness=1,
                        spikecolor="lightgray"
                    ),
                    yaxis=dict(
                        showspikes=False, #turns off horizontal line

                    )
                                        )

                # Update hovertemplate for the 'Paid' trace
                line_fig.update_traces(
                    selector={'name': 'paid'},
                    hovertemplate='<span style="font-size: 18px;">' +
                                'Paid: %{y:.2f}' +
                                '<extra></extra>'
                    )

                # Update hovertemplate for the 'Free' trace
                line_fig.update_traces(
                    selector={'name': 'free'},
                    hovertemplate='<span style="font-size: 18px;">' +
                                'Free: %{y:.2f}' +
                                '<extra></extra>',
        
                )

                st.plotly_chart(line_fig)

    with col_table[1]:
        
        col_free, col_paid, col_line = st.columns(3)
        with col_paid:
            with st.container(border=True):
                # paid songs charts
                st.subheader(paid_title)
                paid_songs_df = engine.top_paid_songs(df=cleaned_listen, state=selected_state)

                chart_paid_songs = alt.Chart(paid_songs_df).mark_bar().encode(
                    x=alt.X('listens:Q', title='Listens'),
                    y=alt.Y('song:N', sort='-x', title=None),
                    tooltip=['song', 'listens']
                ).properties(
                    width=700,
                    height=400,
                ).configure_axis(
                    labelFontSize=14 
                )
                st.altair_chart(chart_paid_songs, use_container_width=True)            

        with col_free:
            with st.container(border=True):
                # free songs chart
                st.subheader(free_title)
                free_songs_df = engine.top_free_songs(df=cleaned_listen, state=selected_state)
                
                chart_free_songs = alt.Chart(free_songs_df).mark_bar().encode(
                    x=alt.X('listens:Q', title='Listens'),
                    y=alt.Y('song:N', sort='-x', title=None),
                    tooltip=['song', 'listens']
                ).properties(
                    width=700,
                    height=400,
                ).configure_axis(
                    labelFontSize=14 
                )
                st.altair_chart(chart_free_songs, use_container_width=True)
        
            with col_line:
                with st.container(border=True):
                    pie_df = engine.create_subscription_pie_chart(df=cleaned_listen, state=selected_state)

                    # Calculate the percentage column based on 'count' column
                    total = pie_df["count"].sum()  # Calculate total count
                    pie_df["percentage"] = (pie_df["count"] / total) * 100  # Calculate percentage

                    # Create a Pandas DataFrame for the side table
                    percentage_df = pie_df[["subscription", "percentage"]].copy()
                    percentage_df["percentage"] = percentage_df["percentage"].map("{:.1f}%".format)  # Format percentage
                    percentage_df = percentage_df.rename(columns={"subscription": "Subscription", "percentage": "Percentage"})  # Rename columns

                    # Create the pie chart
                    chart = alt.Chart(pie_df).mark_arc(outerRadius=120).encode(
                        theta=alt.Theta(field="count", type="quantitative"),
                        color=alt.Color(field="subscription", type="nominal",
                                        scale=alt.Scale(domain=['free', 'paid'],
                                                        range=['orange', 'blue']),
                                        legend=alt.Legend(orient="bottom")),
                        order=alt.Order(field="count", sort="descending"),
                        tooltip=[
                            "subscription",
                            "count",
                            alt.Tooltip("percentage", format=".1f", title="Percentage (%)")
                        ]
                    ).properties(
                        title=pie_title
                    ).configure_view(
                        fillOpacity=0
                    ).configure_title(
                        fontSize=20,
                        font="Arial",
                        anchor="middle",  # Centered title
                        offset=0        
                    )

                    # Display chart
                    st.altair_chart(chart, use_container_width=True)

                    # Display styled table below the chart
                    styled_pie_table = percentage_df[['Subscription', 'Percentage']].style.hide(axis="index").set_table_styles([
                        {'selector': 'td', 'props': [('font-size', '20px'), ('text-align', 'left')]},
                        {'selector': 'th', 'props': [('font-size', '20px'), ('text-align', 'left')]}
                    ])
                    st.markdown(styled_pie_table.to_html(), unsafe_allow_html=True)