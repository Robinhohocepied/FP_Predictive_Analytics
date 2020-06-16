# -------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                      #
#                                                   Deloitte Belgium                                                   #
#                                   Gateway building Luchthaven Nationaal 1J 1930 Zaventem                             #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #
#
# Author list (Alphabetical Order):
#    Name                                       Username                                     Email
#    Ugo Leonard                                uleonard                                     uleonard@deloitte.com
# -------------------------------------------------------------------------------------------------------------------- #
# ###                                               Program Description                                            ### #
# -------------------------------------------------------------------------------------------------------------------- #
# Aim is to create a program that is similar to Prevedere product.
# The program takes as input a given variable (such as sales of a product, P&L line item) and checks for the correlation
# with external variables.
# The external variables are retrieved from db.nomics.world, which is a database of open-source macroeconomic variables
# published by public actors such as NBB, AMECO, WTO, UN, ...
# The script outputs a csv file reporting the correlation between the variable of interest and external variables.
# The correlation is for time t, as well as up to t-12.
# Variables in the file are only those that have a correlation > 0.5 for at least a year.
# -------------------------------------------------------------------------------------------------------------------- #
#                                             Parameters & Libraries                                                   #
# -------------------------------------------------------------------------------------------------------------------- #
import os
import logging
import json, requests
import pandas as pd
import numpy as np
from dbnomics import fetch_series
from joblib import Parallel, delayed

pd.set_option('display.width', 320)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

# Global parameters
data_dir = '../Data/'
results_dir = '../Results/'
rev_csv = data_dir + 'volumes_prepped.csv'
pipe_csv = data_dir + 'Opportunity duration analysis.xlsx'
providers_url = 'https://api.db.nomics.world/v22/providers?limit=10000&offset=0'

logging_file = results_dir + 'logging_file.log'
correlation_csv = results_dir + 'correlation.csv'
nsrcorr_csv = results_dir + 'nsr_corr.csv'

# Query Parameters
nb_rows = 20
var_name = ['1038AG.SPA INTENSE 4*6X 500PET///HOME']
corr_threshold = 0.1
country = ['Belgium']
country_code = ['BE']
eu_list = ['EU', 'World', 'UN']
africa_list = ['Africa', 'World', 'West Africa', 'UN']
# -------------------------------------------------------------------------------------------------------------------- #
#                                             Functions                                                                #
# -------------------------------------------------------------------------------------------------------------------- #

def read_providers(url_providers, country_code, region_list = None):
    '''
    For all countries, fetch the providers that have information about it.
    :param url_providers: URL of the website where to fetch the data (db.nomics.world)
    :param country_code: Country code(s) of the countries we are interested in
    :param region_list: If interested by a region, can also add the region list
    :return: a list of all providers that have information about a certain country/region
    '''
    if region_list is None:
        region_list = []

    flat_list = [item for sublist in [country_code, region_list] for item in sublist]

    response = requests.get(url_providers)
    data = json.loads(response.content)

    providers_list = [item['code'] for item in data['providers']['docs'] if item['region'] in flat_list]

    logging.info('There are {} providers for {}.'.format(len(providers_list), country_code))
    return providers_list


def read_datasets(providers):
    '''
    Creates a dictionary indicating all available datasets available per provider
    :param providers: list of providers from function read_providers()
    :return: a dictionary containing all datasets per provider
    '''
    datasets_dict = {}

    for provider in providers:
        url = 'https://api.db.nomics.world/v22/datasets/{}?limit=500&offset=0'.format(provider)
        response = requests.get(url)
        data = json.loads(response.content)
        try:
            series_dict = {provider: {item['code']: item['name'] for item in data['datasets']['docs']}}
        except:
            logging.info(data['message'])
            continue
        datasets_dict.update(series_dict)

    datasets_count = sum(len(v) for v in datasets_dict.values())

    logging.info('There are {} datasets.'.format(datasets_count))
    return datasets_dict


def remove_duplicate_series(df_gb):
    '''
    If there are series that are available monthly/quarterly and annually, only keep the most granular one
    :param df_gb: a dataframe containing the same variables with potentially multiple frequencies
    :return: a dataframe with only a single frequency
    '''

    frequencies = df_gb['@frequency'].unique()

    if len(frequencies) > 1:
        if 'monthly' in frequencies:
            return df_gb[df_gb['@frequency'] == 'monthly']
        elif 'quarterly' in frequencies:
            return df_gb[df_gb['@frequency'] == 'quarterly']
        else:
            return df_gb[df_gb['@frequency'] == 'annual']
    else:
        return df_gb


def make_monthly(df, start_date, end_date):
    '''
    Sometimes the data is not in the monthly format, make this change here. At the moment, use forward fill.
    TBD: use interpolation (linear or other) to have other values
    :param df: dataset containing all external variables. In this input, the external variables are concatenated
    :param start_date: start date of variable of interest
    :param end_date: end date of variable of interest
    :return: returns a dataframe with monthly values. The external variables are unstacked and each of them is a
             separate column
    '''
    try:
        df = df.copy()
        df_dates = pd.DataFrame({'period': pd.date_range(start=start_date, end=end_date, freq='M')})

        df['series'] = df['series_code'].str.split('.').str[:-1].apply('.'.join)

        df_single = df.groupby('series').apply(remove_duplicate_series)

        df_pivot = df_single.pivot_table(index=['provider_code', 'dataset_code', 'period'],
                                         columns='series_code',
                                         values='value', aggfunc=np.sum).reset_index()

        df_monthly = pd.merge(df_dates.assign(date=df_dates['period'].dt.to_period('M')),
                              df_pivot.assign(date=df_pivot['period'].dt.to_period('M')),
                              how='left', on='date')
        df_monthly = df_monthly.drop(['period_x', 'period_y'], axis=1)
        cols_to_ffill = ['provider_code', 'dataset_code']
        df_monthly[cols_to_ffill] = df_monthly[cols_to_ffill].ffill()
        df_monthly[df_monthly.columns.difference(cols_to_ffill+['date'])] =\
            df_monthly[df_monthly.columns.difference(cols_to_ffill+['date'])].interpolate('linear')

        return df_monthly
    except:
        logging.info('Could not convert dataset {} to monthly data.'.format(df['series_code'].first()))
        return None


def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0, applied on datay
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if isinstance(datax, pd.DataFrame):
        datax = datax.iloc[:,0]

    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return abs(datax.corr(shiftedy))
    else:
        return abs(datax.corr(datay.shift(lag)))


def getSeries(provider, dataset, df_variable, start_date, end_date, flat_list):
    '''
    Fetch the serie from the website and compute correlation between the series and the variable of interest
    :param provider: Provider of the dataset, eg: AMECO, WTO, NBB, ...
    :param dataset: Name of the dataset
    :param df_variable: Variable of interest in the format date - value
    :param start_date: start date of the variable of interest
    :param end_date: end date of the variable of interest
    :param flat_list: country list
    :return: returns a dataset containing the correlation between the variable of interest and all external variables
             in the dataset
    '''
    try:
        df_series = fetch_series(provider_code=provider,
                                 dataset_code=dataset,
                                 max_nb_series=10000)
        df_series.columns = map(str.lower, df_series.columns)

        if set(df_series['@frequency']) == {'annual'}:
            return pd.DataFrame()
        else:
            pass

        if 'country' in df_series.columns:
            df_series['country'] = df_series['country'].str.lower()
            df_series = df_series[df_series['country'].isin(flat_list)]

        df_mapping = df_series[['provider_code', 'series_code', 'dataset_name', 'series_name']].drop_duplicates()
        df_series = df_series[['provider_code', 'dataset_code', 'series_code', 'period', 'value', '@frequency']]
        df_series = df_series[(df_series['period'] >= start_date) &
                              (df_series['period'] <= end_date)]

        if not df_series.empty:
            df_variable = df_variable.reset_index(drop=True)
            df_monthly = make_monthly(df_series, start_date, end_date)
            df_all = pd.merge(df_variable, df_monthly, how='outer', on='date')
            df_all = df_all.sort_values('date').reset_index(drop=True)
            df_all = df_all[df_all.count(axis=1) > 1]

            cols = df_all.columns.difference(['date', 'provider_code', 'dataset_code'])
            cols2 = list(set(df_series['series_code']))
            df_all[cols] = df_all[cols].pct_change().fillna(0).replace([np.inf, -np.inf], 0)
            corr = [crosscorr(df_all[var_name], df_all[var], i) for i in range(0, 13) for var in cols2]

            df_corr = pd.DataFrame(data={
                'series_code': list(np.repeat(cols2, 13)),
                'Time': list(range(0, 13)) * len(cols2),
                'Correlation': corr})

            df_corr = df_corr[df_corr['Correlation'] > corr_threshold]
            df_all = pd.merge(df_corr, df_mapping, how='left', on='series_code')
            df_indicators_corr = df_all[['provider_code', 'series_code', 'dataset_name', 'series_name',
                                         'Time', 'Correlation']]

            df_vals = df_monthly.T
            df_vals.columns = df_vals.iloc[0]

            df_return = pd.merge(df_indicators_corr, df_vals, how='left', left_on='series_code', right_index=True)

            return df_return
    except:
        logging.info("Dataset {} not parsed in getSeries.".format(dataset))


def read_series(df_variable, providers_series, country_code, country, cores):
    '''
    Fetch data from the provider's website and compute the correlation between the various series and the variable
    of interest.
    :param df_variable: dataframe containing the variable of interest
    :param providers_series: dictionnary containing all series for the selected providers
    :param country_code: country code list
    :param country: country list
    :param cores: number of cores, for parallelization
    :return: returns a dataframe containing the correlation between the variable of interest and all series found
    '''
    flat_list = [item.lower() for sublist in [country_code, country] for item in sublist]

    df_indicators_corr = pd.DataFrame({})

    # Start date is a year before the first data point.
    # This is for lag in later stage
    start_date = df_variable['date'].min() - pd.DateOffset(years=1)
    end_date = df_variable['date'].max()

    df_variable['date'] = df_variable['date'].dt.to_period('M')
    for provider in providers_series.keys():
        logging.info('Provider {} started, there are {} datasets'.format(provider,
                                                                         len(providers_series[provider].keys())))
        df_corr = Parallel(n_jobs=cores, verbose=10)(delayed(getSeries)(provider, dataset, df_variable,
                                                                        start_date, end_date,
                                                                        flat_list)
                                         for dataset in providers_series[provider].keys())
        df_indicators_corr = df_indicators_corr.append(pd.concat(df_corr))

        # keep lines below in case of debugging
        #for dataset in providers_series[provider].keys():
        #    df_corr = getSeries(provider, dataset, df_variable, start_date, end_date, flat_list)

    df_indicators_corr = df_indicators_corr.sort_values('Correlation')
    df_indicators_corr = df_indicators_corr[:nb_rows]
    return df_indicators_corr

# -------------------------------------------------------------------------------------------------------------------- #
#                                                 Main                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    num_cores = os.cpu_count()
    print('Number of cores: {}'.format(num_cores))
    logging.basicConfig(filename=logging_file, level=logging.INFO, filemode='w')
    np.random.seed(0)
    df_fte = pd.read_csv(rev_csv, sep='|', thousands=',')
    df_fte['date'] = pd.to_datetime(df_fte['date'], format='%Y-%m-%d')

    df_fte.index = df_fte['date']

    #df_pipe = pd.read_excel(pipe_csv, sheet_name=0)
    #df_pipe = df_pipe[df_pipe['Sales Organization'] == 'Deloitte Consulting']
    #df_pipe['date'] = pd.to_datetime(df_pipe['Creation Date'])
    #df_pipe_gb = df_pipe.groupby(pd.Grouper(key='date', freq='MS'))['Expected Total Value'].agg('sum')
    #df_pipe_gb = pd.merge(df_fte, df_pipe_gb, how='left', left_index=True, right_index=True)
    #
    #df_nsr_corr = pd.DataFrame(columns=['lag', 'correlation'])
    #for lag in range(0, 12):
    #    df_nsr_corr = df_nsr_corr.append({"lag": lag,
    #                                      "correlation": crosscorr(df_pipe_gb['Expected Total Value'],
    #                                                               df_pipe_gb['revenues'],
    #                                                               lag=lag)},
    #                                     ignore_index=True)
    #
    #df_nsr_corr.to_csv(nsrcorr_csv, header=True, index=False, sep='|')
    ## Get the list of providers

    providers_dict = read_providers(providers_url, country_code, eu_list)
    datasets_dict = read_datasets(providers_dict)
    temp_dict = {key: datasets_dict[key] for key in datasets_dict if key in ('AMECO', 'NBB')}
    temp_dict['AMECO'] = dict(list(temp_dict['AMECO'].items())[:15])
    temp_dict['NBB'] = dict(list(temp_dict['NBB'].items())[:5])

    df_corr = read_series(df_fte, temp_dict, country_code, country, num_cores)
    df_corr.to_csv(correlation_csv, header=True, index=False, sep='|')
