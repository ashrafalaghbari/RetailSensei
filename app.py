import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.io as pio
from read_datasets import Datasets
from read_datasets import Datasets
from BuildTsModel import evaluate_forecast

import streamlit as st


# Set the page layout
st.set_page_config(layout="wide", page_title = "Real-Time Dashboard", page_icon = "Active")

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)



st.markdown(
    """
    <style>
    body {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Remove the white space at the top of the page
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

# read the data
datasets = Datasets()
train, test, entire_data, cma, seasonal_indices_series, residuals, \
    seasonal_indices_df, hw_forecast_dev_mean, \
    hw_forecast_dev_lower, hw_forecast_dev_upper, hw_forecast_mean, \
          hw_forecast_lower, hw_forecast_upper, hw_forecast  = datasets.get_datasets()


# Update the dashboard with the latest data
if len(test) > len(hw_forecast_dev_mean):
    diff = len(test) - len(hw_forecast_dev_mean)
    last_date = hw_forecast_dev_mean.index[-1]
    next_dates = []

    for i in range(1, diff+1):
        next_date = last_date + pd.DateOffset(months=i)
        next_dates.append(next_date)

    next_values = hw_forecast_mean.loc[next_dates]
    hw_forecast_dev_mean = pd.concat([hw_forecast_dev_mean, pd.Series(next_values, index=next_values.index)])

tab1, tab2 = st.tabs(["Monitor", "Analysis"])

with tab1:

    # a1 = st.columns(1)
    # # Place the first plot in the first column
    # with a1:
    # Row A
    # Create a trace for the seasonal indices
    fig = px.line(entire_data, x=entire_data.index, y=entire_data, title='<b>Real Advance Retail Sales in US: Retail Trade</b>')
    fig.update_xaxes(title_text='<b>Month</b>', tickfont=dict(family='Arial', size=12))
    fig.update_yaxes(title_text='<b>Billions of 2023 Dollars</b>', tickfont=dict(family='Arial', size=12))
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(title=dict(x=0.5, y=0.9, font=dict(size=20)))
    fig.update_yaxes(tickfont_family="Arial Black")
    fig.update_xaxes(tickfont_family="Arial Black")
    fig.update_traces(hovertemplate='Month: %{x}<br>Sales: %{y:.2f}<extra></extra>')

    fig.add_trace(go.Scatter(
        x=hw_forecast_mean.index,
        y=hw_forecast_mean,
        mode='lines',
        line=dict(color='black', width=2),
        name='<b>Forecast</b>',
        hovertemplate='Month: %{x}<br>Sales: %{y:.2f}<extra></extra>'
    ))


    fig.add_trace(go.Scatter(
        x=test["2023-05-01":].index,
        y=test["2023-05-01":],
        mode='markers',
        marker=dict(symbol='star', color='purple', size=10),
        name='<b>Actual</b>',
        hovertemplate='Month: %{x}<br>Sales: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=hw_forecast_upper.index,
        y=hw_forecast_upper,
        mode='lines',
        line=dict(color='blue'),
        name='<b>95% Confidence Interval</b>',
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line_shape='spline',
        hovertemplate='Month: %{x}<br>Max. Sales: %{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=hw_forecast_lower.index,
        y=hw_forecast_lower,
        mode='lines',
        line=dict(color='blue'),
        showlegend=False,
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.2)',
        line_shape='spline',
        hovertemplate='Month: %{x}<br>Min. Sales: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text='<b>Real Advance Retail Sales in US: Retail Trade</b>',
            x=0.35,
            y=0.85,
            font=dict(size=20)
        )
    )

    fig.update_layout(legend=dict(
        font=dict(size=12),
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="right",
        x=0.6
    ))
    fig.update_layout(height=400, width=1200, showlegend=True, margin=dict(t=160, b=0, l=0, r=0))
    st.plotly_chart(fig)
        # a1,a2, a3, a4, a5 = st.columns(5, gap="small")
    st.markdown("""<h3 style='text-align: center;font-family:Arial Black; 
                font-size: 17.5px ;'>Performance Metrics</h3>""", 
                    unsafe_allow_html=True)
   

    b1, b2, b3, b4, b5, b6, b7, b8, b9 = st.columns(9)

    last_models_metrics = evaluate_forecast('Holt-Winters Model', test[:-1], hw_forecast_dev_mean[:-1])
    current_models_metrics = evaluate_forecast('Holt-Winters Model', test[:], hw_forecast_dev_mean[:])
    # with a1:  
    #     # Add an empty line before the plot to push it to the top of the page
    #     st.write("")
    #     st.plotly_chart(fig3)

        # Row b



    st.markdown("""
        <style>
        [data-testid=stVerticalBlock] {
        gap: 0.9rem;
        }
        [data-testid=stHorizontalBlock] > div {
            padding-right: 0.1rem !important;
            padding-left: 0.1rem !important;
        }
        </style>
        """, unsafe_allow_html=True)



    with b1:
    # Render the metrics in a nice grey background
        # add the title performance metrics
    

        # Define the custom CSS style
        custom_style = """
        <style>
        .dashboard {
            background-color: #f2f2f2;
            padding: 10px;
            border-radius: 8px;
        }
        .metric {
            margin-bottom: 8px;
            font-size: 20px;
            font-family: Arial Black, sans-serif;
        }
        </style>
        """

        # Render the custom CSS
        st.markdown(custom_style, unsafe_allow_html=True)

        # Render the metrics in the dashboard layout
        st.markdown("""
        <div class="dashboard">
            <div class="metric" style="text-align: center; font-family: Helvetica ; font-size: 10px;">
                Last date:{}
            </div>
            <div class="metric", style='text-align: center; font-family: Arial Black; font-size: 16px'>
                Last MAPE
            </div>
        <div class="metric" style="text-align: center; color: blue; font-family: Arial Black; font-size: 16px">
                <b>{}%</b>
            </div>
        </div>
        """.format(test.index.date[-2], last_models_metrics['MAPE'].round(2)), unsafe_allow_html=True)

    with b2:
    # Render the metrics in a nice grey background
        # add the title performance metrics

        # Render the custom CSS
        st.markdown(custom_style, unsafe_allow_html=True)

        # Chnage the color of the metric based on the performance
        if current_models_metrics['MAPE'] < last_models_metrics['MAPE']:
            color = 'green'
        else:
            color = 'red'

        # Render the metrics in the dashboard layout
        st.markdown("""
        <div class="dashboard">
            <div class="metric" style="text-align: center;font-family: Arial; font-size: 10px;">
                Updated: {}
            </div>
            <div class="metric", style='text-align: center; font-family: Arial Black; font-size: 16px'>
                Current MAPE
            </div>
        <div class="metric" style="text-align: center; color: {}; font-family: Arial Black; font-size: 16px">
                <b>{}%</b>
            </div>
        </div>
        """.format(test.index.date[-1],
                    color, current_models_metrics['MAPE'].round(2)), unsafe_allow_html=True)
        
    with b3:
        # Render the metrics in a nice grey background
        # Render the custom CSS
        st.markdown(custom_style, unsafe_allow_html=True)

        # Render the metrics in the dashboard layout
        st.markdown("""
        <div class="dashboard">
            <div class="metric" style="text-align: center; font-family: Helvetica ; font-size: 10px;">
                Last date:{}
            </div>
            <div class="metric", style='text-align: center; font-family: Arial Black; font-size: 16px'>
                Last RMSE
            </div>
        <div class="metric" style="text-align: center; color: blue; font-family: Arial Black; font-size: 16px">
                <b>{}</b>
            </div>
        </div>
        """.format(test.index.date[-2],
                last_models_metrics['RMSE'].round(2)), unsafe_allow_html=True)
        
    with b4:
        # Render the metrics in a nice grey background
        # Render the custom CSS
        st.markdown(custom_style, unsafe_allow_html=True)
            # Chnage the color of the metric based on the performance
        if current_models_metrics['RMSE'] < last_models_metrics['RMSE']:
            color = 'green'
        else:
            color = 'red'

        # Render the metrics in the dashboard layout
        st.markdown("""
        <div class="dashboard">
            <div class="metric" style="text-align: center;font-family: Arial; font-size: 10px;">
                Updated: {}
            </div>
            <div class="metric", style='text-align: center; font-family: Arial Black; font-size: 16px'>
                Current RMSE
            </div>
        <div class="metric" style="text-align: center; color: {};; font-family: Arial Black; font-size: 16px">
                <b>{}</b>
            </div>
        </div>
        """.format(test.index.date[-1], 
                color,
                current_models_metrics['RMSE'].round(2)), unsafe_allow_html=True)
        
    with b5:
        # Render the metrics in a nice grey background
        # Render the custom CSS
        st.markdown(custom_style, unsafe_allow_html=True)

        # Render the metrics in the dashboard layout
        st.markdown("""
        <div class="dashboard">
            <div class="metric" style="text-align: center; font-family: Helvetica ; font-size: 10px;">
                Last date:{}
            </div>
            <div class="metric", style='text-align: center; font-family: Arial Black; font-size: 16px'>
                Last MAE
            </div>
        <div class="metric" style="text-align: center; color: blue;; font-family: Arial Black; font-size: 16px">
                <b>{}</b>
            </div>
        </div>
        """.format(test.index.date[-2],
                    last_models_metrics['MAE'].round(2)), unsafe_allow_html=True)
        
    

    with b6:
        # Render the metrics in a nice grey background
        # Render the custom CSS
        st.markdown(custom_style, unsafe_allow_html=True)
        # Chnage the color of the metric based on the performance
        if current_models_metrics['MAE'] < last_models_metrics['MAE']:
            color = 'green'
        else:
            color = 'red'

        # Render the metrics in the dashboard layout
        st.markdown("""
        <div class="dashboard">
            <div class="metric" style="text-align: center;font-family: Arial; font-size: 10px;">
                Updated: {}
            </div>
            <div class="metric", style='text-align: center; font-family: Arial Black; font-size: 16px'>
                Current MAE
            </div>
        <div class="metric" style="text-align: center; color: {};; font-family: Arial Black; font-size: 16px">
                <b>{}</b>
            </div>
        </div>
        """.format(test.index.date[-1],
                color,
                current_models_metrics['MAE'].round(2)), unsafe_allow_html=True)

    with b7:
        # Render the metrics in a nice grey background
        # Render the custom CSS
        st.markdown(custom_style, unsafe_allow_html=True)

        # Render the metrics in the dashboard layout
        st.markdown("""
        <div class="dashboard">
            <div class="metric" style="text-align: center;font-family: Arial; font-size: 10px;">
            Updated: {}
            </div>
            <div class="metric", style='text-align: center; font-family: Arial Black; font-size: 16px'>
                Actual
            </div>
        <div class="metric" style="text-align: center; color: blue; font-family: Arial Black; font-size: 16px">
                <b>{}</b>
            </div>
        </div>
        """.format(test.index.date[-1], test.iloc[-1].round(2)), unsafe_allow_html=True)

    with b8:
    # Render the metrics in a nice grey background
        # Render the custom CSS
        st.markdown(custom_style, unsafe_allow_html=True)

        # Render the metrics in the dashboard layout
        st.markdown("""
        <div class="dashboard">
            <div class="metric" style="text-align: center; font-family: Helvetica ; font-size: 10px;">
                Updated: {}
            </div>
            <div class="metric", style='text-align: center; font-family: Arial Black; font-size: 16px'>
                Forecast
            </div>
            <div class="metric" style="text-align: center; color: blue;; font-family: Arial Black; font-size: 16px">
                <b>{}</b>
            </div>
        </div>
        """.format(test.index.date[-1],
                hw_forecast_dev_mean[-1].round(2)), unsafe_allow_html=True)

 

    with b9:
        # Get the last date in the test set
        last_date = test.index[-1]

        # Generate the next date
        next_date = pd.date_range(start=last_date, periods=2, freq='MS')[1]
        # Render the metrics in a nice grey background
        # Render the custom CSS
        st.markdown(custom_style, unsafe_allow_html=True)

        # Render the metrics in the dashboard layout
        st.markdown("""
        <div class="dashboard">
            <div class="metric" style="text-align: center; font-family: Helvetica ; font-size: 10px;">
                Next date: {}
            </div>
            <div class="metric", style='text-align: center; font-family: Arial Black; font-size: 16px'>
                Next Forecast
            </div>
        <div class="metric" style="text-align: center; color: blue;; font-family: Arial Black; font-size: 16px">
                <b>{}</b>
            </div>
        </div>
        """.format(next_date.date(),
                hw_forecast_mean[next_date].round(2)), unsafe_allow_html=True)



with tab2:
    st.markdown("""<p style='text-align: center;font-family:Arial Black; 
            font-size: 25px ;'>A closer look into the data</p>""", 
                unsafe_allow_html=True)
    
    # Display information about the performance metrics
    st.markdown("## Performance Metrics")
    st.markdown("")
    st.markdown("### Root Mean Squared Error (RMSE)")
    st.markdown("")
    st.markdown("The Root Mean Squared Error (RMSE) is a commonly used metric to measure the average deviation between the predicted and actual values. It provides an overall indication of the forecast accuracy, taking into account both the magnitude and direction of the errors. The RMSE is measured in the same units as the forecasted variable, which in this case is in billions of dollars.")
    st.markdown("")
    st.markdown("### Mean Absolute Percentage Error (MAPE)")
    st.markdown("")
    st.markdown("The Mean Absolute Percentage Error (MAPE) is a metric that measures the average percentage deviation between the predicted and actual values. It provides a relative measure of the forecast accuracy and is particularly useful when comparing forecasts across different time periods or datasets. The MAPE is expressed as a percentage.")
    st.markdown("")
    st.markdown("### Mean Absolute Error (MAE)")
    st.markdown("")
    st.markdown("The Mean Absolute Error (MAE) is a metric that measures the average absolute deviation between the predicted and actual values. It provides a simple and interpretable measure of the forecast accuracy, representing the average magnitude of the errors. The MAE is measured in the same units as the forecasted variable, which in this case is in billions of dollars.")
    st.markdown("")
    st.markdown("Please keep in mind that these metrics serve as indicators of forecast accuracy, and it's important to consider them in conjunction with other factors when assessing the quality of the forecasts.")
    st.markdown("")
    # Create the columns with unequal width
    c1 = st.columns(1)

    # Display the decomposed plots

    # Add the explanation about classical decomposition
    st.markdown("""
        <h3 style='font-size: 20px; text-align: center;'>Classical Decomposition</h3>
        
        <p>The classical decomposition is a method used to break down a time series into its underlying components, namely the trend,
        seasonal indices, and residuals. This decomposition helps to understand the individual patterns and variations within the data.</p>
        
        <p><b>Trend:</b> The trend component represents the long-term, systematic changes or patterns in the data. It captures the overall
        direction in which the series is moving, regardless of the seasonal or irregular fluctuations. The trend provides insights into
        the underlying growth or decline of the phenomenon being observed.</p>
        
        <p><b>Seasonal component:</b> The seasonal indices reveal the systematic, recurring patterns or fluctuations that occur within a
        specific time period, such as daily, weekly, monthly, or yearly. These indices measure the deviation from the average for each
        corresponding period. By examining the seasonal indices, one can identify the seasonal effects and their impact on the series.
        This information is valuable for forecasting and understanding the seasonality of the data.</p>
        
        <p><b>Residuals:</b> The residuals, also known as the irregular component or noise, represent the unexplained variation in the
        data after removing the trend and seasonal effects. These residuals capture the random or unpredictable fluctuations that cannot
        be accounted for by the trend and seasonality. Analyzing the residuals helps to identify any remaining patterns or anomalies
        in the data that are not explained by the trend and seasonality.</p>
        
        <p>By decomposing a time series into its components, analysts can gain insights into the different sources of variation,
        make informed decisions based on the individual components, and develop more accurate forecasts by modeling the trend, seasonality,
        and residuals separately.</p>
    """, unsafe_allow_html=True)


    # Create a subplot figure with three subplots
    fig2 = sp.make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0, horizontal_spacing=0)

    # Subplot 1: Training data and trend component
    fig2.add_trace(go.Scatter(x=train.index, y=train, name='<b>Train</b>', line=dict(color='red'),
                            hovertemplate='Month: %{x}<br>Sales: %{y}'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=cma.index, y=cma, name='<b>Trend-CMA</b>', line=dict(color='black'),
                            hovertemplate='Month: %{x}<br>Sales: %{y}'), row=1, col=1)

    # Subplot 2: Seasonal component
    fig2.add_trace(go.Scatter(x=seasonal_indices_series.index, y=seasonal_indices_series,
                            name='<b>Seasonal Component</b>', line=dict(color='green', width=2),
                            hovertemplate='Month: %{x}<br>Sales: %{y}'), row=2, col=1)

    # Subplot 3: Residuals component
    fig2.add_trace(go.Scatter(x=residuals.index, y=residuals, name='<b>Residuals</b>', line=dict(color='purple'),
                            hovertemplate='Month: %{x}<br>Sales: %{y}'), row=3, col=1)
    fig2.add_trace(go.Scatter(x=residuals.index, y=residuals.rolling(window=12).mean(),
                            name='<b>12 Rolling Mean</b>', line=dict(color='orange'),
                            hovertemplate='Month: %{x}<br>Sales: %{y}'), row=3, col=1)
    fig2.add_trace(go.Scatter(x=residuals.index, y=residuals.rolling(window=12).std(),
                            name='<b>12 Rolling Std</b>', line=dict(color='blue'),
                            hovertemplate='Month: %{x}<br>Sales: %{y}'), row=3, col=1)

    # Customize the layout and axes titles for subplot figure
    fig2.update_layout(
        title='<b>Retail Sales Decomposition</b>',
        title_font=dict(size=20),
        title_x=0.3,  # Center the title horizontally
        xaxis=dict(title=''),
        yaxis=dict(title='', domain=[0.65, 1]),
        yaxis2=dict(title='<b>Billions of 2023 Dollars</b>', domain=[0.35, 0.6]),
        yaxis3=dict(title='', domain=[0, 0.25]),
    )
    # Add legends to the subplots
    fig2.add_trace(go.Scatter(name='<b>Train</b>'), row=1, col=1)
    fig2.add_trace(go.Scatter(name='<b>Trend-CMA</b>'), row=1, col=1)
    fig2.add_trace(go.Scatter(name='<b>Seasonal Component</b>'), row=2, col=1)
    fig2.add_trace(go.Scatter(name='<b>Residuals</b>'), row=3, col=1)
    fig2.add_trace(go.Scatter(name='<b>12 Rolling Mean</b>'), row=3, col=1)
    fig2.add_trace(go.Scatter(name='<b>12 Rolling Std</b>'), row=3, col=1)

    # Adjust the subplot spacing
    fig2.update_layout(height=500, width=1200, showlegend=True, margin=dict(t=100, b=0, l=0, r=0))
    # Adjust the legend position
    fig2.update_layout(legend=dict(
        font=dict(size=12),
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="right",
        x=0.75
    ))
    # adjust the layout of the title
    fig2.update_layout(title=dict(x=0.4, y=0.95))
    # Set the x-axis title at the bottom
    fig2.update_xaxes(title='<b>Month</b>', row=3, col=1)
    fig2.update_yaxes(tickfont_family="Arial Black")
    fig2.update_xaxes(tickfont_family="Arial Black")
    st.plotly_chart(fig2)


    c2 = st.columns(1)
       #Add  explanation about seasonal indices
    st.markdown("""
        
        <p>The seasonal indices provide valuable insights into the comparison between each season and the yearly average,
        indicating whether values are lower or higher by a specific amount or percentage.</p>
        
        <p>For instance, the seasonal index for January reveals a decrease of $46.64 billion, indicating that January sales
        are below the yearly average.</p>
        
        <p>On the other hand, the seasonal index for December shows an impressive increase of $77.40 billion, representing
        a significant rise of 2397.7% compared to the previous month. This remarkable surge suggests that December sales
        surpass the yearly average, outperforming all other months. This analysis aligns perfectly with the expected pattern,
        as sales typically experience a surge towards the end of the year due to Christmas and other major sales events.
        Consequently, sales observe a notable decline as people reduce their purchasing activities after the holiday season.</p>
        
        <p>This information helps to understand the seasonal patterns and trends in sales, allowing for better decision-making
        and planning in the retail industry.</p>""", unsafe_allow_html=True)

    # Set the template to ggplot2
    pio.templates.default = "ggplot2"

    # Create a trace for the seasonal indices
    trace = go.Scatter(x=list(range(1, len(seasonal_indices_df.iloc[:,0]) + 1)),
                    y=seasonal_indices_df.iloc[:,0],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Seasonal Indices',
                    text=[f"{round(index, 2)}" for index in seasonal_indices_df.iloc[:,0]])

    # Create a trace for the annotations
    annotations = []
    for index, row in seasonal_indices_df.iterrows():
        annotations.append(dict(x=index + 1, y=row['seasonal_indices'], text=f"{row['seasonal_indices']}",
                                font=dict(size=10)))

    # Create a layout for the plot
    layout = go.Layout(title=dict(text="<b>Seasonal Indices in Billions of Dollars for Retail Sales in US</b> ", font=dict(size=15)),
                    xaxis=dict(title=dict(text='<b>Month</b>', font=dict(size=12)),
                                tickfont=dict(size=15),
                                dtick=1),
                    yaxis=dict(title=dict(text='<b>Billons of 2023 Dollars</b>',font=dict(size=12))),
                    hovermode='x',
                    hoverlabel=dict(bgcolor='white', font=dict(size=12)),
                    margin=dict(l=50, r=50, t=80, b=50),
                    legend=dict(x=0.05, y=0.95, bgcolor='white', bordercolor='gray', borderwidth=1, font=dict(size=12)))

    # Create a figure and plot the data
    fig3 = go.Figure(data=[trace], layout=layout)
    # Update the y-axis label font family of fig3
    fig3.update_yaxes(title_font=dict(family="Arial Black"), tickfont_family="Arial Black")
    fig3.update_xaxes(title_font=dict(family="Arial Black"), tickfont_family="Arial Black")
    fig3.update_xaxes(range=[1, 12])
    fig3.update_layout(
        title=dict(
            x=0.3,
            y=1,
            font=dict(size=20)
        )
    )

    def hovertext(value, percent):
        if value > 0:
            return f"<b>Change: <span style='color:green'>{value} ({percent}%)</span></b>"
        else:
            return f"<b>Change: <span style='color:red'>{value} ({percent}%)</span></b>"

    # fig.update_traces(hovertemplate='%{y:.2f}<extra></extra>')
    fig3.update_traces(text=[hovertext(value, percent) if pd.notna(value) else 'N/A' 
                            for value, percent in zip(seasonal_indices_df.iloc[:,1], seasonal_indices_df.iloc[:,2])])
    fig3.update_layout(height=500, width=1200, margin=dict(t=50, b=0, l=0, r=0))
    # Display the seasonal indices graph
    st.plotly_chart(fig3)

    import streamlit as st

    # Display the table

    # Add the explanation about the components
    st.markdown("""
        <h3 style='font-size: 20px; text-align: center;'>Holt-Winters Forecast </h3>
        
        <p>The Holt-Winters model is a popular method for forecasting time series data that incorporates trends and seasonality.
        The forecast table consists of four components: forecasted values, standard error, lower 95% confidence interval (CI),
        and upper 95% confidence interval (CI). These components provide valuable information for understanding and utilizing the forecast.</p>
        
        <p><b>Forecasted Values:</b> The forecasted values represent the predicted values for the future time periods based on the Holt-Winters model.
        These values indicate the expected trend and seasonality patterns in the data and can be used for making future predictions.</p>
        
        <p><b>Standard Error:</b> The standard error measures the accuracy or uncertainty of the forecasted values. It quantifies the average
        amount by which the forecasted values may deviate from the actual values. A smaller standard error indicates a more reliable forecast.</p>
        
        <p><b>Lower 95% Confidence Interval (CI):</b> The lower 95% confidence interval provides a range within which the actual values are
        likely to fall with 95% confidence. It represents the lower bound of the uncertainty range around the forecasted values.</p>
        
        <p><b>Upper 95% Confidence Interval (CI):</b> The upper 95% confidence interval represents the upper bound of the uncertainty range
        around the forecasted values. It indicates the upper limit within which the actual values are likely to fall with 95% confidence.</p>
        
        <p>By analyzing these components, analysts can assess the reliability of the forecasts, evaluate the uncertainty associated with the predictions,
        and make informed decisions based on the range of potential outcomes. The confidence intervals provide a measure of the forecast accuracy
        and can be used to assess the risk and variability in the future values.</p>
    """, unsafe_allow_html=True)


    # Show table for hw_forecast
    table_forecast = hw_forecast.copy()
    table_forecast = table_forecast.rename(columns={"forecast_mean": "Forecast",
                                                    "forecast_se": "Standard Error",
                                                    "lower_ci_0.95": "Lower 95% CI", 
                                                    "upper_ci_0.95": "Upper 95%"})
    table_forecast.index.name = "Date"
    table_forecast.index = table_forecast.index.date
    

    # Center the table
    st.markdown("<h2 style='font-size: 25px; text-align: left;'>Forecasted values and 95% confidence interval</h2>", unsafe_allow_html=True)
    styles = [{'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}]
    table_html = table_forecast.style.set_table_styles(styles).render()
    # st.markdown("<div style='text-align: center;'>{}</div>".format(table_html), unsafe_allow_html=True)
    st.markdown("<div style='text-align: center;'><div>{}</div></div>".format(table_html), unsafe_allow_html=True)
    st.write("\n")



















