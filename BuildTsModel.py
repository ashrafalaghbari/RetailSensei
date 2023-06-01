import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters.results import HoltWintersResultsWrapper
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox as ljung
from statsmodels.stats.stattools import jarque_bera as jb
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def forecast_holt_winters(
        model, 
        horizon, 
        seasonal_period, 
        significance_level=0.05,
  ):
    """
    Forecast future values using the Holt-Winters method with additive trend and seasonality.
    Calculates the forecast mean, standard error, and confidence intervals.
    ETS(A,A_d,A)
    using the following equation:
    sigma_h^2 = sigma^2 * [1 + alpha^2(h-1) + gamma*k(2*alpha + gamma)
                          + (beta*phi*h*(2*alpha*(1-phi) + beta*phi)) / ((1-phi)^2)
                          - (beta*phi*(1-phi^h)*(2*alpha*(1-phi^2) + beta*phi*(1+2*phi-phi^h))) / ((1-phi)^2 * (1-phi^2))
                          + (2*beta*gamma*phi*(k*(1-phi^m) - phi^m*(1-phi^(mk)))) / ((1-phi)*(1-phi^m))]
    ETS(A,A_d,A) is a state space model for time series data. The model has three components: the error component (A),
    the trend component (A_d), and the seasonal component (A). The model is used to forecast future values of
    a time series based on its past values. 

    The equation shown is used to calculate the variance of the forecast errors for the ETS(A,A_d,A) model.
    The equation takes in several parameters, including the smoothing parameters (alpha, beta, gamma, and phi),
    the forecast horizon (h), and the number of seasons in a year (m). 

    The equation is broken down into several components, each of which represents a different aspect of the model.
    The first component represents the variance of the errors in the data. The second component represents the variance
    of the trend component. The third component represents the variance of the seasonal component.
    The fourth component represents the covariance between the trend and seasonal components.
    The fifth component represents the covariance between the errors and the seasonal component. 

    The equation is used to calculate the variance of the forecast errors,
    which is a measure of how accurate the model's forecasts are likely to be.
    A lower variance indicates that the model's forecasts are likely to be more accurate.

    Parameters
    ----------
    model : statsmodels.tsa.holtwinters.HoltWintersResultsWrapper
        Fitted Holt-Winters model.
    horizon : int
        Number of periods to forecast.
    seasonal_period : int
        Number of periods in a seasonal cycle.
    alpha : float, optional
        Significance level for calculating confidence intervals. Default is 0.05.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the forecast mean, standard error, and confidence intervals.

    Raises
    ------
    ValueError
        If horizon or seasonal_period are not positive integers.
    TypeError
        If model is not a HoltWintersResultsWrapper object.


    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: principles and practice, 2nd edition,

    """
    # Check input parameters
    if not isinstance(model, HoltWintersResultsWrapper):
        raise TypeError("model must be a HoltWintersResultsWrapper object")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(seasonal_period, int) or seasonal_period <= 0:
        raise ValueError("seasonal_period must be a positive integer")

    # Define the model parameters
    h = horizon
    m = seasonal_period
    k = (h-1)/m   
    alpha = model.params['smoothing_level']
    beta = model.params['smoothing_trend']
    gamma = model.params['smoothing_seasonal']
    phi = model.params['damping_trend']

    # Calculate residuals variance
    residuals_var = np.mean(model.resid**2)

    forecast_var = np.array([], dtype=np.float64)

    for h in range(1, h+1):
        component1 = 1 + alpha**2*(h-1)
        component2 = gamma*k*(2*alpha + gamma)
        component3 = (beta*phi*h*(2*alpha*(1-phi) + beta*phi))/((1-phi)**2)
        component4 = (beta*phi*(1-phi**h)*(2*alpha*(1-phi**2) + beta*phi*(1+2*phi-phi**h)))/((1-phi)**2*(1-phi**2))
        component5 = (2*beta*gamma*phi*(k*(1-phi**m) - phi**m*(1-phi**(m*k))))/((1-phi)*(1-phi**m))
        # Calculate forecasting variance for h-step ahead forecast
        forecast_var_h= residuals_var  * (component1 + component2 + component3 - component4 + component5)

        forecast_var = np.append(forecast_var, forecast_var_h)

    # Calculate the standard error of the forecast
    forecast_se = np.sqrt(forecast_var)

    # Calculate the z value
    percentile_point = 1 - significance_level / 2
    # Calculate the number of observations
    n = len(model.resid)
    # Percentile point of a Student's t-distribution with n-p degrees of freedom
    z_value = stats.t.ppf(percentile_point, df=n-1)

    # Calculate the margin of error
    margin_of_error = z_value * forecast_se

    # Calculate the forecast
    predicted_mean = model.forecast(h)
    # Calculate the confidence interval
    lower_ci = predicted_mean - margin_of_error
    upper_ci = predicted_mean + margin_of_error

    # Return the forecast, the confidence interval, and the margin of error in a DataFrame
    return pd.DataFrame({'forecast_mean': predicted_mean, 
                         'forecast_se': forecast_se,
                         f'lower_ci_{1-significance_level}': lower_ci, 
                         f'upper_ci_{1-significance_level}': upper_ci})


def residcheck(residuals, lags):
    
    """
    
    Function to check if the residuals are white noise. Ideally the residuals should be uncorrelated, zero mean, 
    constant variance and normally distributed. First two are must, while last two are good to have. 
    If the first two are not met, we have not fully captured the information from the data for prediction. 
    Consider different model and/or add exogenous variable. 
    
    If Ljung Box test shows p> 0.05, the residuals as a group are white noise. Some lags might still be significant. 
    
    Lags should be min(2*seasonal_period, T/5) where T is the length of the time series.
    
    Function taken from https://pawarbi.github.io/blog/forecasting/r/python/rpy2/altair/fbprophet/
    ensemble_forecast/uncertainty/simulation/2020/04/21/timeseries-part2.html#Confidence-Interval-vs.-Prediction-Interval
    plots from: https://tomaugspurger.github.io/modern-7-timeseries.html
    
    """
    resid_mean = np.mean(residuals)
    lj_p_val = np.mean(ljung(x=residuals, lags=lags).iloc[:,1])
    norm_p_val =  jb(residuals)[1]
    adfuller_p = adfuller(residuals)[1]
    
    
    
    fig = plt.figure(figsize=(10,8))
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2);
    acf_ax = plt.subplot2grid(layout, (1, 0));
    kde_ax = plt.subplot2grid(layout, (1, 1));

    residuals.plot(ax=ts_ax)
    plot_acf(residuals, lags=lags, ax=acf_ax);
    sns.kdeplot(residuals);
    #[ax.set_xlim(1.5) for ax in [acf_ax, kde_ax]]
    sns.despine()
    plt.tight_layout();
    
    print("** Mean of the residuals: ", np.around(resid_mean,2))
    
    print("\n** Ljung Box Test, p-value:", np.around(lj_p_val,3), "(>0.05, Uncorrelated)" if (lj_p_val > 0.05) else "(<0.05, Correlated)")
    
    print("\n** Jarque Bera Normality Test, p_value:", np.around(norm_p_val,3), "(>0.05, Normal)" if (norm_p_val>0.05) else "(<0.05, Not-normal)")
    
    print("\n** AD Fuller, p_value:", np.around(adfuller_p,3), "(>0.05, Non-stationary)" if (adfuller_p > 0.05) else "(<0.05, Stationary)")
    
    
    
    return ts_ax, acf_ax, kde_ax


def plot_forecast(
        train: pd.Series = None, 
        test: pd.Series = None, 
        fitted_values: pd.Series = None, 
        forecasted_values: pd.DataFrame = None, 
        model_name: str = None, 
        get_conf_int: bool = False,
        significance_level: float = 0.95, 
        only_forecast_period: bool = False,
        legend_loc: str = 'best',
        title: str = "Real Advance Retail Sales in US: Retail Trade",
        xlabel: str = 'Month',
        ylabel: str = 'Billions of 2023 Dollars',
        **kwargs
  )->None:
    """
    Plots the original data, fitted values, test data, and forecasts.

    Parameters
    ----------
    train : pd.Series
        Training data.
    test : pd.Series
        Test data.
    fitted_values : pd.Series
        Fitted values.
    forecasted_values : pd.DataFrame
        Forecasted values.
    model_name : str
        Name of the model.
    get_conf_int : bool, optional
        Whether to get the confidence interval, by default False
    significance_level : float, optional
        Significance level for the confidence interval, by default 0.95
    only_forecast_period : bool, optional
        Whether to plot only the forecast period, by default False
    legend_loc : str, optional
        Location of the legend, by default 'best'
    title : str, optional
        Title of the plot, by default None
    xlabel : str, optional
        Label for the x-axis, by default 'Month'
    ylabel : str, optional
        Label for the y-axis, by default 'Billions of 2023 Dollars'
    **kwargs : optional
        Additional keyword arguments to pass to the plot function.

    Returns
    -------
    None

    """
    # Plot the original data, fitted values, test data, and forecasts
    if not only_forecast_period:
        plt.plot(train.index, train, label='Train', color='blue')
        plt.plot(fitted_values.index, fitted_values, label='Fitted values', color='red')
    if test is not None:   
        plt.plot(test.index, test, label='Test', color='green')
    plt.plot(forecasted_values.index, forecasted_values['forecast_mean'], label='Forecast', color='black')

    if get_conf_int:
        plt.fill_between(forecasted_values.index, 
                         forecasted_values[f'lower_ci_{significance_level}'], 
                         forecasted_values[f'upper_ci_{significance_level}'], 
                         color='blue', 
                         label=f'{int(significance_level*100)}% Confidence Interval',  
                         alpha=0.2)
    plt.legend(loc=legend_loc, **kwargs)

    # Set label properties
    if title:
        plt.title(f'{title} {model_name}', color='black', weight='bold')
    plt.setp(plt.gca().get_xticklabels(), color='black', weight='bold')
    plt.setp(plt.gca().get_yticklabels(), color='black', weight='bold')
    plt.xlabel(xlabel, color='black', weight='bold')
    plt.ylabel(ylabel, color='black', weight='bold')

    # Bold the legend
    plt.setp(plt.gca().get_legend().get_texts(), weight='bold')

    # # Show the plot
    plt.show()

    

   



def evaluate_forecast(
        model: str, 
        test_values: pd.Series,
        forecast_values: pd.Series 
  ) -> dict:
    
    """
    Evaluate the forecast using MAE, RMSE, and MAPE.
    
    Parameters:
    model (str): The name of the forecasting model.
    forecast_values (array-like): The forecasted values.
    test_values (array-like): The actual values.
    
    Returns:
    A dictionary containing the MAE, RMSE, and MAPE.
    """
    # Calculate the MAE, RMSE, and MAPE
    mae = mean_absolute_error(test_values, forecast_values)
    rmse = np.sqrt(mean_squared_error(test_values, forecast_values))
    mape = np.mean(np.abs((test_values - forecast_values) / test_values)) * 100

    
    # Return the  results as a dictionary
    return {'Model':model, 'MAE': np.round(mae, 4), 
                           'RMSE': np.round(rmse,4), 
                           'MAPE': np.round(mape, 4)}

def calculate_pi_difference(test: pd.Series, forecast: pd.Series) -> dict:
    """
    Calculates the difference between the upper and lower bounds of the prediction interval.

    Parameters
    ----------
    test : pd.Series
        Test data.
    forecast : pd.Series
        Forecasted values.
    alpha : float, optional
        Significance level for the prediction interval, by default 0.05.

    Returns
    -------
    dict
        A dictionary with the coverage proportion, coverage description, confidence interval width, and confidence interval width description.

    """

    # Calculate the coverage proportion
    coverage = 1 - ((test < forecast.iloc[:, 0]) | (test > forecast.iloc[:, 1])).mean() 

    # Calculate the difference between the upper and lower bounds of the prediction interval
    pi_width =  forecast.iloc[:, 1] - forecast.iloc[:, 0]


    # Create a dictionary with the results
    results = {
        'coverage': coverage.round(3),
        'coverage_description': 'The proportion of true values within the confidence interval.',
        'ci_width': pi_width.sum().round(3),
        'ci_width_description': 'The width of the confidence interval, which indicates the uncertainty of the forecast.'
    }

    return results