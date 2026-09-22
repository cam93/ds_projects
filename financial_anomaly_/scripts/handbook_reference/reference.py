"""Official Handbook reference functions, isolated from notebook execution.

Source revision: 81cf7d1714bb7b2f5b496407d9055d91dc68dc25
Source: https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
GPL-3.0; see adjacent LICENSE. Functions extracted with ast.unparse; logic unchanged.
"""
import datetime
import random
import numpy as np
import pandas as pd

def generate_customer_profiles_table(n_customers, random_state=0):
    np.random.seed(random_state)
    customer_id_properties = []
    for customer_id in range(n_customers):
        x_customer_id = np.random.uniform(0, 100)
        y_customer_id = np.random.uniform(0, 100)
        mean_amount = np.random.uniform(5, 100)
        std_amount = mean_amount / 2
        mean_nb_tx_per_day = np.random.uniform(0, 4)
        customer_id_properties.append([customer_id, x_customer_id, y_customer_id, mean_amount, std_amount, mean_nb_tx_per_day])
    customer_profiles_table = pd.DataFrame(customer_id_properties, columns=['CUSTOMER_ID', 'x_customer_id', 'y_customer_id', 'mean_amount', 'std_amount', 'mean_nb_tx_per_day'])
    return customer_profiles_table

def generate_terminal_profiles_table(n_terminals, random_state=0):
    np.random.seed(random_state)
    terminal_id_properties = []
    for terminal_id in range(n_terminals):
        x_terminal_id = np.random.uniform(0, 100)
        y_terminal_id = np.random.uniform(0, 100)
        terminal_id_properties.append([terminal_id, x_terminal_id, y_terminal_id])
    terminal_profiles_table = pd.DataFrame(terminal_id_properties, columns=['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id'])
    return terminal_profiles_table

def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    x_y_customer = customer_profile[['x_customer_id', 'y_customer_id']].values.astype(float)
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))
    available_terminals = list(np.where(dist_x_y < r)[0])
    return available_terminals

def generate_transactions_table(customer_profile, start_date='2018-04-01', nb_days=10):
    customer_transactions = []
    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))
    for day in range(nb_days):
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)
        if nb_tx > 0:
            for tx in range(nb_tx):
                time_tx = int(np.random.normal(86400 / 2, 20000))
                if time_tx > 0 and time_tx < 86400:
                    amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    if amount < 0:
                        amount = np.random.uniform(0, customer_profile.mean_amount * 2)
                    amount = np.round(amount, decimals=2)
                    if len(customer_profile.available_terminals) > 0:
                        terminal_id = random.choice(customer_profile.available_terminals)
                        customer_transactions.append([time_tx + day * 86400, day, customer_profile.CUSTOMER_ID, terminal_id, amount])
    customer_transactions = pd.DataFrame(customer_transactions, columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])
    if len(customer_transactions) > 0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions['TX_TIME_SECONDS'], unit='s', origin=start_date)
        customer_transactions = customer_transactions[['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']]
    return customer_transactions

def add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df):
    transactions_df['TX_FRAUD'] = 0
    transactions_df['TX_FRAUD_SCENARIO'] = 0
    transactions_df.loc[transactions_df.TX_AMOUNT > 220, 'TX_FRAUD'] = 1
    transactions_df.loc[transactions_df.TX_AMOUNT > 220, 'TX_FRAUD_SCENARIO'] = 1
    nb_frauds_scenario_1 = transactions_df.TX_FRAUD.sum()
    print('Number of frauds from scenario 1: ' + str(nb_frauds_scenario_1))
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(n=2, random_state=day)
        compromised_transactions = transactions_df[(transactions_df.TX_TIME_DAYS >= day) & (transactions_df.TX_TIME_DAYS < day + 28) & transactions_df.TERMINAL_ID.isin(compromised_terminals)]
        transactions_df.loc[compromised_transactions.index, 'TX_FRAUD'] = 1
        transactions_df.loc[compromised_transactions.index, 'TX_FRAUD_SCENARIO'] = 2
    nb_frauds_scenario_2 = transactions_df.TX_FRAUD.sum() - nb_frauds_scenario_1
    print('Number of frauds from scenario 2: ' + str(nb_frauds_scenario_2))
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(n=3, random_state=day).values
        compromised_transactions = transactions_df[(transactions_df.TX_TIME_DAYS >= day) & (transactions_df.TX_TIME_DAYS < day + 14) & transactions_df.CUSTOMER_ID.isin(compromised_customers)]
        nb_compromised_transactions = len(compromised_transactions)
        random.seed(day)
        index_fauds = random.sample(list(compromised_transactions.index.values), k=int(nb_compromised_transactions / 3))
        transactions_df.loc[index_fauds, 'TX_AMOUNT'] = transactions_df.loc[index_fauds, 'TX_AMOUNT'] * 5
        transactions_df.loc[index_fauds, 'TX_FRAUD'] = 1
        transactions_df.loc[index_fauds, 'TX_FRAUD_SCENARIO'] = 3
    nb_frauds_scenario_3 = transactions_df.TX_FRAUD.sum() - nb_frauds_scenario_2 - nb_frauds_scenario_1
    print('Number of frauds from scenario 3: ' + str(nb_frauds_scenario_3))
    return transactions_df

def is_weekend(tx_datetime):
    weekday = tx_datetime.weekday()
    is_weekend = weekday >= 5
    return int(is_weekend)

def is_night(tx_datetime):
    tx_hour = tx_datetime.hour
    is_night = tx_hour <= 6
    return int(is_night)

def get_customer_spending_behaviour_features(customer_transactions, windows_size_in_days=[1, 7, 30]):
    customer_transactions = customer_transactions.sort_values('TX_DATETIME')
    customer_transactions.index = customer_transactions.TX_DATETIME
    for window_size in windows_size_in_days:
        SUM_AMOUNT_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(str(window_size) + 'd').sum()
        NB_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(str(window_size) + 'd').count()
        AVG_AMOUNT_TX_WINDOW = SUM_AMOUNT_TX_WINDOW / NB_TX_WINDOW
        customer_transactions['CUSTOMER_ID_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)
        customer_transactions['CUSTOMER_ID_AVG_AMOUNT_' + str(window_size) + 'DAY_WINDOW'] = list(AVG_AMOUNT_TX_WINDOW)
    customer_transactions.index = customer_transactions.TRANSACTION_ID
    return customer_transactions

def get_count_risk_rolling_window(terminal_transactions, delay_period=7, windows_size_in_days=[1, 7, 30], feature='TERMINAL_ID'):
    terminal_transactions = terminal_transactions.sort_values('TX_DATETIME')
    terminal_transactions.index = terminal_transactions.TX_DATETIME
    NB_FRAUD_DELAY = terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').sum()
    NB_TX_DELAY = terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').count()
    for window_size in windows_size_in_days:
        NB_FRAUD_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(str(delay_period + window_size) + 'd').sum()
        NB_TX_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(str(delay_period + window_size) + 'd').count()
        NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
        NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY
        RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW
        terminal_transactions[feature + '_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)
        terminal_transactions[feature + '_RISK_' + str(window_size) + 'DAY_WINDOW'] = list(RISK_WINDOW)
    terminal_transactions.index = terminal_transactions.TRANSACTION_ID
    terminal_transactions.fillna(0, inplace=True)
    return terminal_transactions

def get_train_test_set(transactions_df, start_date_training, delta_train=7, delta_delay=7, delta_test=7, sampling_ratio=1.0, random_state=0):
    train_df = transactions_df[(transactions_df.TX_DATETIME >= start_date_training) & (transactions_df.TX_DATETIME < start_date_training + datetime.timedelta(days=delta_train))]
    test_df = []
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)
    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()
    for day in range(delta_test):
        test_df_day = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training + delta_train + delta_delay + day]
        test_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training + delta_train + day - 1]
        new_defrauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD == 1].CUSTOMER_ID)
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)
        test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]
        test_df.append(test_df_day)
    test_df = pd.concat(test_df)
    if sampling_ratio < 1:
        train_df_frauds = train_df[train_df.TX_FRAUD == 1].sample(frac=sampling_ratio, random_state=random_state)
        train_df_genuine = train_df[train_df.TX_FRAUD == 0].sample(frac=sampling_ratio, random_state=random_state)
        train_df = pd.concat([train_df_frauds, train_df_genuine])
    train_df = train_df.sort_values('TRANSACTION_ID')
    test_df = test_df.sort_values('TRANSACTION_ID')
    return (train_df, test_df)

def card_precision_top_k_day(df_day, top_k):
    df_day = df_day.groupby('CUSTOMER_ID').max().sort_values(by='predictions', ascending=False).reset_index(drop=False)
    df_day_top_k = df_day.head(top_k)
    list_detected_compromised_cards = list(df_day_top_k[df_day_top_k.TX_FRAUD == 1].CUSTOMER_ID)
    card_precision_top_k = len(list_detected_compromised_cards) / top_k
    return (list_detected_compromised_cards, card_precision_top_k)

def card_precision_top_k(predictions_df, top_k, remove_detected_compromised_cards=True):
    list_days = list(predictions_df['TX_TIME_DAYS'].unique())
    list_days.sort()
    list_detected_compromised_cards = []
    card_precision_top_k_per_day_list = []
    nb_compromised_cards_per_day = []
    for day in list_days:
        df_day = predictions_df[predictions_df['TX_TIME_DAYS'] == day]
        df_day = df_day[['predictions', 'CUSTOMER_ID', 'TX_FRAUD']]
        df_day = df_day[df_day.CUSTOMER_ID.isin(list_detected_compromised_cards) == False]
        nb_compromised_cards_per_day.append(len(df_day[df_day.TX_FRAUD == 1].CUSTOMER_ID.unique()))
        (detected_compromised_cards, card_precision_top_k) = card_precision_top_k_day(df_day, top_k)
        card_precision_top_k_per_day_list.append(card_precision_top_k)
        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(detected_compromised_cards)
    mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()
    return (nb_compromised_cards_per_day, card_precision_top_k_per_day_list, mean_card_precision_top_k)
