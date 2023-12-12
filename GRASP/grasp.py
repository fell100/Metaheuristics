import multiprocessing
from pprint import pprint
import pandas as pd
import yfinance as yf
import math
import datetime
import random
import numpy as np
import pickle
from multiprocessing import Pool
import logging
import os
import time

def load_dictionary(filename):
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


sharpe_ratio_dict = load_dictionary('sharpe_ratio_dict.pkl')

sharpe_df = pd.read_parquet('sharpe_df.parquet')

def greedy_portfolio(RLC_length = 150, portfolio_length = 15):

    portfolio = []
    RLC = sorted(sharpe_ratio_dict.items(), key=lambda x: x[1], reverse=True)[:RLC_length]

    while len(portfolio) < portfolio_length:

        random_stock = random.choice(RLC)
        portfolio.append(random_stock)

        RLC.pop(RLC.index(random_stock))
    return portfolio

def generate_random_solution(portfolio_length , bin_str_size = 7):
    bin_string = ''
    for letter in range(portfolio_length):
        binary_letter_string = "".join(str(np.random.randint(0,2)) for i in range(bin_str_size))
        bin_string = bin_string + binary_letter_string
    return bin_string

def decode_binary_to_weights(solution, bin_str_size = 7):
    while True:
        try:
            weight_list = []
            for pos in range(0, len(solution), bin_str_size):
                binary_weight = str(solution[pos: pos+bin_str_size])
                weight = (int(binary_weight, 2))
                weight_list.append(weight)
                sum_w = np.sum(weight_list)
            return [(weight / sum_w) for weight in weight_list]
        except:
            continue


def portfolio_returns_sharpe(weights, df, pf_returns = True, sharpe = True):

    days = len(set([time_index[1] for time_index in df.index.values]))

    df['weight'] = df.index.get_level_values('ticker').map(weights)

    # Calculate daily returns
    df['return'] = df.groupby('ticker')['Adj Close'].pct_change()

    # Calculate portfolio return
    df['portfolio_return'] = df['return'] * df['weight']

    # Calculate daily portfolio return
    daily_portfolio_return = df.groupby('time')['portfolio_return'].sum()

    # Calculate cumulative portfolio return
    cumulative_portfolio_return = (1 + daily_portfolio_return).cumprod() - 1
    
    # Calculate daily risk-free rate (e.g., use Treasury yields)
    risk_free_rate = 0.02 / days  # Assuming a 2% annual risk-free rate

    # Calculate portfolio Sharpe ratio
    daily_sharpe_ratio = (daily_portfolio_return.mean() - risk_free_rate) / daily_portfolio_return.std()

    # Calculate annualized Sharpe ratio
    annual_sharpe_ratio = daily_sharpe_ratio * (days ** 0.5)

    
    if pf_returns == True and sharpe == True:
        return cumulative_portfolio_return, annual_sharpe_ratio
    elif pf_returns == True and sharpe == False:
        return cumulative_portfolio_return
    elif sharpe == True and pf_returns == False:
        return annual_sharpe_ratio
    else:
       return None 


def perturbate_solution(s, n = 1):
    for i in range(n):
        pos = np.random.choice(len(s))
        
        binary_list = list(s)
        
        if binary_list[pos] == '0':
            binary_list[pos] = '1'
        else:
            binary_list[pos] = '0'
        
        s = ''.join(binary_list)
    
    return s

def update_tabu_list(tabu_list, max_length, solution):
    if max_length == len(tabu_list):
        tabu_list.pop()
        tabu_list.insert(0, solution)
    else:
        tabu_list.append(solution)
        
    return tabu_list

def generate_neighborhood(n_neighbors, current_solution, bin_str_size):
    neighborhood = []
    for i in range(n_neighbors):
        new_solution = current_solution
        while((new_solution in neighborhood) or (new_solution == current_solution)):
            new_solution = perturbate_solution(current_solution, 1)
        neighborhood.append([decode_binary_to_weights(new_solution, bin_str_size), new_solution])

    return neighborhood


def tabu_search(max_iterations, n_neighbors, early_stop_n, tabu_list_len, s0, df, bin_str_size):

    tabu_list, bin_list, bin_bests, score_list, best_scores, mean_scores = [], [], [], [], [], []

    stocks_names = [stock for stock in s0[0].keys()]
    portfolio_df = df.loc[stocks_names]

    best_score = portfolio_returns_sharpe(s0[0], portfolio_df, False) * (-1)
    best_solution = s0[0]
    iteration, best_iteration = 0, 0
    
    while (iteration <= max_iterations) and (iteration - best_iteration <= early_stop_n):

        nb = generate_neighborhood(n_neighbors, s0[1], bin_str_size)
        
        neighborhood = [neighbor[1] for neighbor in nb]

        neighborhood_weights = [neighbor[0] for neighbor in nb]

        neighbors_scores = []
        for idx_n in range(len(neighborhood)): 
            temp_dict = {key: value for key, value in zip(stocks_names, neighborhood_weights[idx_n])}

            score = portfolio_returns_sharpe(temp_dict, portfolio_df, False)

            neighbors_scores.append(- score)

        min_neighbor_score =  min(neighbors_scores)
        best_neighbor = neighborhood[neighbors_scores.index(min_neighbor_score)]
        best_neighbor_w = neighborhood_weights[neighbors_scores.index(min_neighbor_score)]
        

        
        if (best_neighbor not in tabu_list) or (min_neighbor_score <= best_score):
            tabu_list = update_tabu_list(tabu_list, tabu_list_len, best_neighbor)
            s0, s0_score = [best_neighbor_w, best_neighbor], min_neighbor_score

            bin_list.append(s0)
            score_list.append(s0_score)
            mean_scores.append(np.mean(neighbors_scores))
            if s0_score <= best_score:
                best_score = s0_score
                best_solution = s0
                best_iteration = iteration
                bin_bests.append(best_solution)
                best_scores.append(best_score)
            

        iteration += 1
        
    return best_solution, best_score, bin_list, score_list, bin_bests, best_scores, mean_scores



def greedy_solution(RLC_length = 250, portfolio_length = 15, bin_str_size = 5):
    #Initial greedy solutin
    portfiolio = greedy_portfolio(RLC_length, portfolio_length)

    portfiolio = [ticker[0] for ticker in portfiolio]

    solution = generate_random_solution(len(portfiolio), bin_str_size)

    portfolio_weights = decode_binary_to_weights(solution, bin_str_size)

    #greedy solution with random weights
    portfolio_dict = {k : v for (k, v) in zip(portfiolio, portfolio_weights)}
    s0 = [portfolio_dict, solution]
    return s0


max_iter = 32
RLC_length = 250
portfolio_length = 30
bin_str_size = 5

best_results = []




# Define a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("tabu_search_log.txt")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def run_iteration(iter_num):
    s0 = greedy_solution(RLC_length, portfolio_length, bin_str_size)
    tabu_run = tabu_search(max_iterations=300, n_neighbors=30, early_stop_n=15,
                            tabu_list_len=15, s0=s0, df=sharpe_df, bin_str_size=bin_str_size)
    best_solution, best_score, bin_list, score_list, bin_bests, best_scores, mean_scores = tabu_run
    result_dict = {k: v for (k, v) in zip([key for key in s0[0].keys()], best_solution[0])}

    # Log iteration results
    logger.info(f"Iteration {iter_num + 1}:")
    logger.info(f"Best solution: {result_dict}")
    logger.info(f"Best score: {-best_score}")

    best_results.append([result_dict, -best_score, score_list, mean_scores, bin_list])
 
    time.sleep(np.random.rand() * 100)
    
    csv_file_path = 'results.csv'
    
    if not os.path.exists(csv_file_path):
        df = pd.DataFrame(columns=['result_dict', 'best_solution', 'best_score', 'score_list', 'mean_score', 'bin_list'])
    else:
        df = pd.read_csv(csv_file_path)

    new_data = {'result_dict': result_dict,'best_solution' : best_solution, 'best_score': -best_score, 'score_list': score_list, 'mean_score': mean_scores, 'bin_list' : bin_list}
    #df = df.append(new_data, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    # Use all available CPU cores
    with Pool(processes=None) as pool:
        pool.map(run_iteration, range(max_iter))

    # Save best results
    with open("best_results.pkl", "wb") as f:
        pickle.dump(best_results, f)

    logger.info("Best results saved to best_results.pkl")