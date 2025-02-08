"""
Author: Akshay Juyal (Department of Computer Science GSU)
Date: March 19, 2024
Description: This script performs preprocessing of covid amino acid sequence data or
            similar RNA based viral sequences as well as 
            calculate the fitness value for each sequences.
"""
import os
import pickle
import random
import copy
import argparse
import shutil
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
from scipy.stats import binom
from scipy.special import comb


def preprocess_data(input, datepath, outpath):
    """preprocessing step"""
    # find values till a given date

    # read the full file
    df = pd.read_csv(input, header=None)
    df = df.replace('e0',0) # apparently there is a value in columns one which are 'e0'
    df = df.astype(np.int8)
    input_file = os.path.basename(input).split(".")[0]
    #add dates to the data 
    with open(datepath, "r") as filestream:
        for val in filestream:
            dates = val.split(",")
        df["dates"]=pd.to_datetime(dates)
      
    # sort wrt dated
    df = df.sort_values(by = 'dates') 

    # save sorted sequence file
    with open(os.path.join(outpath,f'{input_file}_sorted_with_dates.pkl'), 'wb') as file:
        pickle.dump(df,file, protocol=5)
    
    print("*"*50)
    print(f"The input has been processed and saved at {outpath}")
    print("*"*50)

def shuffle_snp(rows_count, columns_count, shuffled_matrix):
    for col_index in tqdm(range(columns_count)): # traverse all columns
        col_snps = shuffled_matrix[:, col_index]
        unique_values = np.unique(col_snps)
        if len(unique_values) > 1: #check if all 0's or 1's
            for index in range(rows_count*2):
                pos1, pos2 =  random.sample(range(rows_count),2) # swaping index with a random position
                col_snps[pos1], col_snps[pos2] = col_snps[pos2], col_snps[pos1]
                shuffled_matrix[:, col_index] = col_snps
    return shuffled_matrix

def calculate_pair_statistics(params):
    # idx, in_idx, orig_C1, orig_C2, shfl_C1, shfl_C2 = params
    idx, in_idx, orig_C1, orig_C2, shuffled, ratio = params

    z_z = np.sum((orig_C1 == 0) & (orig_C2 == 0))
    z_o = np.sum((orig_C1 == 0) & (orig_C2 == 1))
    o_z = np.sum((orig_C1 == 1) & (orig_C2 == 0))
    o_o = np.sum((orig_C1 == 1) & (orig_C2 == 1))

    if ratio:
        if shuffled:
            return [1, int(idx), int(in_idx), float((((z_z * o_o)+1)) / (((z_o * o_z)+1))), z_z, z_o, o_z, o_o]
        else:
            return [0, int(idx), int(in_idx), float((((z_z * o_o)+1)) / (((z_o * o_z)+1))), z_z, z_o, o_z, o_o]
    else:
        if shuffled:
            return [1, int(idx), int(in_idx), float(((z_z * o_o)) - (((z_o * o_z)+1))), z_z, z_o, o_z, o_o]
        else:
            return [0, int(idx), int(in_idx), float(((z_z * o_o)) - ((z_o * o_z))), z_z, z_o, o_z, o_o]

def parallel_computation(pair):
    with Pool() as pool:  
        results = list(tqdm(pool.imap(calculate_pair_statistics, pair), total=len(pair), desc="Computing Pair ratios"))
    return results


def create_adj_matrix(columns_count, significant_red_edges, significant_blue_edges):
    adj_matrix = np.zeros([columns_count,columns_count],dtype=np.int8) 
    for red_pair in significant_red_edges:
        adj_matrix[int(red_pair[1]),int(red_pair[2])] = adj_matrix[int(red_pair[2]),int(red_pair[1])] = 1

    for blue_pair in significant_blue_edges:
        adj_matrix[int(blue_pair[1]),int(blue_pair[2])] = adj_matrix[int(blue_pair[2]),int(blue_pair[1])] = -1
    
    return adj_matrix

def create_edge_matrix(columns_count,adj_matrix, outpath, datefilter):
    G, graph = preprocess(adj_matrix, outpath, datefilter, False)
    edge_mat = np.zeros([len(graph.edges()), columns_count],dtype=np.int8) 
    edge_info = [(val[0],val[1],(1 if val[2]["color"]=='red' else -1)) for val in graph.edges.data()]
    for idx, val in enumerate(edge_info):
        edge_mat[idx][val[0]]+=val[2]
        edge_mat[idx][val[1]]+=val[2]
    
    return edge_mat

def preprocess(G, outpath, datefilter, log):
        graph = nx.Graph()
        num_nodes = len(G)
        graph.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                        if G[i][j] == 1:
                                graph.add_edge(i, j, color='red')
                                pass
                        elif G[i][j] == -1:
                                graph.add_edge(i, j, color='blue')
        # print("total edges before removing isolated nodes",len(graph.edges()))
        nodes_with_isolates = len(graph.nodes())
        print("Total nodes before removing isolated nodes",nodes_with_isolates)
        graph.remove_nodes_from(list(nx.isolates(graph)))
        nodes_without_isolates = len(graph.nodes())
        print("Total nodes after removing isolated nodes",nodes_without_isolates)
        print("Total edges after removing isolated nodes",len(graph.edges()))
        if log:
            with open (os.path.join(outpath, datefilter, f"statistics_{datefilter}.log"),"a") as logfile:
                logfile.write(f"Total nodes before removing isolated nodes {nodes_with_isolates}. \n")
                logfile.write(f"Total nodes after removing isolated nodes {nodes_without_isolates}. \n")
                logfile.write(f"Total edges after removing isolated nodes {len(graph.edges())}. \n")
        updated = nx.to_numpy_array(graph,dtype=np.int8)
        node_mapping = {node: new_index for new_index, node in enumerate(graph.nodes())}
        # updated=np.zeros(nx.to_numpy_array(graph,dtype=np.int8).shape,dtype=np.int8)
        for i,j in graph.edges():
                if graph[i][j]['color']=='blue':
                        updated[node_mapping[i]][node_mapping[j]] = -1
                        updated[node_mapping[j]][node_mapping[i]] = -1

        return updated, graph

def display_graph(G, pos, title, outpath, datefilter):

    plt.figure(figsize=(50,50))
    nx.draw(G, pos, with_labels=True, node_color='lightgray', node_size=1500, font_size=15)
    red_edges = [(val[0],val[1]) for val in G.edges.data() if val[2]["color"]=='red']
    blue_edges =  [(val[0],val[1]) for val in G.edges.data() if val[2]["color"]=='blue']
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', width=2)
    nx.draw_networkx_edges(G, pos, edgelist=blue_edges, edge_color='blue', width=2)
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color='black')
    plt.title('"SNP Graph"')
    plt.axis('off')
    plt.savefig(os.path.join(outpath, datefilter, f"epistaticNetwork_{datefilter}.png"), format="PNG")
    # plt.show()
    print(title,"=> Red:",len(red_edges)," blue:",len(blue_edges))
    with open (os.path.join(outpath, datefilter, f"statistics_{datefilter}.log"),"a") as logfile:
            logfile.write(f"There are {len(blue_edges)} significant Blue edges and {len(red_edges)} significant Red edges\n")


def pair_snps(columns_count, matrix, shuffled, ratio):
    pairs=[]
    for idx in tqdm(range(columns_count)):
        for in_idx in range(idx + 1, columns_count):
            orig_C1 = matrix[:, idx]
            orig_C2 = matrix[:, in_idx]
            # shfl_C1 = shuffled_matrix[:, idx]
            # shfl_C2 = shuffled_matrix[:, in_idx]
            # pairs.append((idx, in_idx, orig_C1, orig_C2, shfl_C1, shfl_C2))
            pairs.append((idx, in_idx, orig_C1, orig_C2, shuffled, ratio))
    return pairs
    
def find_significant_edges(combined_p_sorted_red, combined_p_sorted_blue, p_val, shuffles):
    significant_red_edges = []
    significant_blue_edges = []
    shuffled_seen_red = 0
    shuffled_seen_blue = 0
    for idx, val in tqdm(enumerate(combined_p_sorted_red)):
        if val[0] == 0:
            significant_red_edges.append(val)
        if val[0] == 1:
            shuffled_seen_red+=1/shuffles
            # if shuffled_seen_red==2:
            #     break
        if ((shuffled_seen_red)/(idx+1)) >= p_val:
            print(f"idx: {idx+1}, {shuffled_seen_red} ",shuffled_seen_red/(idx+1))
            break
    for idx, val in tqdm(enumerate(combined_p_sorted_blue)):
        if val[0] == 0:
            significant_blue_edges.append(val)
        if val[0] == 1:
            shuffled_seen_blue+=1/shuffles
            # if shuffled_seen_blue==2: 
            #     break
        if ((shuffled_seen_blue)/(idx+1)) >= p_val:
            print(f"idx: {idx+1}, {shuffled_seen_blue} ",shuffled_seen_blue/(idx+1))
            break
    return significant_blue_edges,significant_red_edges

def create_adj_matrix_binom(columns_count,original_p):
    # [0, int(idx), int(in_idx), float((((z_z * o_o)+1)) / (((z_o * o_z)+1))), z_z, z_o, o_z, o_o]

    # add weights p value edges
    # add weights vertices
    # add weights combination
    adj_matrix = np.zeros([columns_count,columns_count],dtype=np.int8)
    for val in tqdm(original_p):
        z_z = val[4]
        z_o = val[5]
        o_z = val[6]
        o_o = val[7]
        thr = 0.05/comb(columns_count,2)
        if (z_z * o_o) > (z_o * o_z) and (o_o >= 1) and (z_o >= 1) and (z_z >= 1):
            p = ((o_z*z_o)+1) / ((z_z*columns_count)+1) # columns_count=1273
            prob = 1 - binom.cdf(o_o,columns_count,p)
            if (prob <= thr):
                adj_matrix[int(val[1]),int(val[2])] = adj_matrix[int(val[2]),int(val[1])] = 1
        elif (z_z * o_o) < (z_o * o_z) and (o_o >= 1) and (z_o >= 1) and (z_o >= 1):
            p = ((o_o*z_z)+1) / ((z_o*columns_count)+1) # columns_count=1273
            prob = 1 - binom.cdf(o_o,columns_count,p)
            if (prob <= thr):
                adj_matrix[int(val[1]),int(val[2])] = adj_matrix[int(val[2]),int(val[1])] = -1   
    return adj_matrix

def main(input, start_date, datefilter, p_val, shuffles, outpath, calc_type, save_files):

    _, file_extension = os.path.splitext(input)
    if not file_extension.lower() == ".pkl":
        print("*"*50,"\n")
        print("\tThis is not a pkl file please use the Preprocessing functionality to generate one\n")
        print("*"*50)
        return

    print("=====================> Reading input file")
    with open(input, 'rb') as file:
        pickle_df = pickle.load(file)
    
    pickle_df_till_date = pickle_df[pickle_df["dates"]<=datefilter]
    pickle_df_till_date_window = pickle_df[(pickle_df["dates"]<=datefilter) & (pickle_df["dates"]>=start_date)]
    original_df =  pickle_df_till_date.drop(["dates"], axis=1)
    original_df_window =  pickle_df_till_date_window.drop(["dates"], axis=1)
    # original_cols = original_df.columns
    if save_files:
        print(f"=====================> saving filtered dataset from {start_date} till {datefilter}")
        pickle_df_till_date.to_csv(os.path.join(outpath, datefilter,f"data_{datefilter}.csv"),index=None)

    original_matrix = original_df.copy().to_numpy()
    original_matrix_window = original_df_window.copy().to_numpy()
    rows_count, columns_count = original_df.shape

    if calc_type in ["ratio", "diff"]:

        shuffled_matrix = original_df.copy().to_numpy()
        print(f"=====================> permuting data {shuffles} times")
        shuffled_matrices = []
        with Pool() as pool:
            shuffled_matrices = pool.starmap(shuffle_snp, [(rows_count, columns_count, shuffled_matrix)] * shuffles)
        
        if calc_type == "ratio": 
            orig_pairs = pair_snps(columns_count, original_matrix, False, True)
            shuffled_pairs = []
            print("=====================> Pairing SNP's for shuffled matrices")
            for mat in shuffled_matrices:
                shuffled_pairs.append(pair_snps(columns_count, mat, True, True))

        else:
            orig_pairs = pair_snps(columns_count, original_matrix, False, False)
            shuffled_pairs = []
            print("=====================> Pairing SNP's for shuffled matrices")
            for mat in shuffled_matrices:
                shuffled_pairs.append(pair_snps(columns_count, mat, True, False))
            
        print("=====================> calculating pairwise statistics for original matices 'SNP's")

        original_p = parallel_computation(orig_pairs)
        shuffled_p_args=[(pairs) for pairs in shuffled_pairs] 
        shuffled_p =[]
        print("=====================> calculating pairwise statistics for shuffled matices 'SNP's")
        for arg in shuffled_p_args:
            shuffled_p.extend(parallel_computation(arg))

        combined_p = np.concatenate((original_p , shuffled_p), axis=0)
        if save_files:
            print(f"=====================> Saving combined statisics for each pair in the original and {shuffles} shuffled matrices")
            np.savetxt(os.path.join(outpath, datefilter, f"combined_pstats_{datefilter}.csv"), combined_p, fmt="%.7f", delimiter=",")
        combined_p_sorted_red = combined_p[combined_p[:, 3].argsort()[::-1]] #decending
        combined_p_sorted_blue = combined_p[combined_p[:, 3].argsort()]
        
        print("=====================> Finding significant edges")
        significant_blue_edges,significant_red_edges = find_significant_edges(combined_p_sorted_red, combined_p_sorted_blue, p_val, shuffles)

        print(f"There are {len(significant_blue_edges)} significant Blue edges and {len(significant_red_edges)} significant Red edges")
        
        adj_matrix = create_adj_matrix(columns_count, significant_red_edges, significant_blue_edges)

    elif calc_type == "binomial":
        print("=====================> Pairing SNP's for matrices")
        orig_pairs = pair_snps(columns_count, original_matrix, False, True)
        print("=====================> calculating pairwise statistics for matices 'SNP's")
        original_p = parallel_computation(orig_pairs)
        print("=====================> Finding significant edges")
        adj_matrix = create_adj_matrix_binom(columns_count, original_p)
    if save_files:
        print("=====================> Saving Adjecency matrix")
        np.savetxt(os.path.join(outpath, datefilter, f"adj_mat_{datefilter}.csv"), adj_matrix, fmt="%d", delimiter=",")
    edge_matrix = create_edge_matrix(columns_count, adj_matrix, outpath, datefilter)
    if save_files:
        print("=====================> Saving Edge matrix")
        np.savetxt(os.path.join(outpath, datefilter, f"edge_mat_{datefilter}.csv"), edge_matrix, fmt="%d", delimiter=",")
    population_matrix = original_matrix_window.copy().transpose()
    fitness_matrix= np.matmul(edge_matrix, population_matrix)
    if save_files:
        print("=====================> Saving Fitness matrix")
        np.savetxt(os.path.join(outpath, datefilter, f"fitness_mat_{datefilter}.csv"), fitness_matrix, fmt="%d", delimiter=",")
    fitness_vector = np.sum(fitness_matrix, axis=0)
    print("=====================> Saving Fitness vector")
    np.savetxt(os.path.join(outpath, datefilter, f"fitnessVector_{datefilter}.csv"), fitness_vector,fmt="%d", delimiter=",")
    min_fitness = min(fitness_vector)
    norm_fitness_vector = [fitness_val - min_fitness if fitness_val!=0 else fitness_val for fitness_val in fitness_vector]
    print("=====================> Saving Normalized Fitness vector")
    np.savetxt(os.path.join(outpath, datefilter, f"norm_fitnessVector_{datefilter}.csv"), norm_fitness_vector, fmt="%d", delimiter=",")

    print("=====================> Generating and saving bar chart")

    fitness_info = dict(Counter(fitness_vector))
    keys = list(fitness_info.keys())
    values = list(fitness_info.values())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

    normalized_values = np.array(values)
    normalized_values_log = np.log(normalized_values + 1)
    dates= list(pickle_df_till_date_window["dates"])

    ax1.bar(keys, normalized_values_log, color='skyblue')
    ax1.set_title('#sequences per fitness value')
    ax1.set_xlabel('fitness Value')
    ax1.set_ylabel('Logarithmic Normalized Values(#sequences)')

    ax2.scatter(fitness_vector, dates)
    ax2.set_title('sequences dates wrt fitness value')
    ax2.set_xlabel('fitness Value')
    ax2.set_ylabel('Dates')
    plt.savefig(os.path.join(outpath, datefilter, f"plot_{datefilter}.png"), format="PNG")

    ordinal_dates = [date.toordinal() for date in dates]
    # Convert ordinal dates to datetime objects
    datetime_dates = [datetime.fromordinal(date) for date in ordinal_dates]

    # Calculate correlation coefficient
    correlation_data = [['edge to date','edge to dist','dist to date','density edge to date']]
    hamming_dist = np.sum(original_matrix_window, axis=1)
    correlation = np.corrcoef(fitness_vector, ordinal_dates)[0, 1]
    correlation_f_h = np.corrcoef(fitness_vector, hamming_dist)[0, 1]
    correlation_date_h = np.corrcoef(ordinal_dates, hamming_dist)[0, 1]
    density_freq = []
    for val in zip(hamming_dist,fitness_vector):
        if val[0]==0:
            density_freq.append(0)
        else:
             density_freq.append(val[1]/val[0])
    correlation_date_density_f = np.corrcoef(ordinal_dates, density_freq)[0, 1]
    with open (os.path.join(outpath, datefilter, f"statistics_{datefilter}.log"),"a") as logfile:
            logfile.write(f"The correlation of fitness value to the sampling date is: {correlation}. \n"
                          f"The correlation of fitness value to the hamming dist is: {correlation_f_h}. \n"
                          f"The correlation of dates to the hamming dist is: {correlation_date_h}. \n"
                          f"The correlation of density fitness value to the sampling date is: {correlation_date_density_f}. \n")
    correlation_data.append([correlation,correlation_f_h,correlation_date_h,correlation_date_density_f])
    with open(os.path.join(outpath, datefilter, f"correlation_{datefilter}.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(correlation_data)
    print("=====================> Generating and saving correlation graph")
    plt.figure(figsize=(20,20))
    plt.scatter(datetime_dates, fitness_vector, color='blue', label=f'Correlation: {correlation:.2f}')
    plt.xlabel('Dates')
    plt.ylabel('Fitness Vector')
    plt.title('Correlation Between Dates and Fitness Vector')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig(os.path.join(outpath, datefilter, f"correlation_{datefilter}.png"), format="PNG")

    print("=====================> Generating and saving Epistatic network graph")
    # graph = nx.Graph()
    g_matrix = copy.deepcopy(adj_matrix)
    g_matrix,G = preprocess(g_matrix, outpath, datefilter, True)
    # num_nodes = len(g_matrix)
    # graph.add_nodes_from(range(num_nodes))
    pos = nx.shell_layout(G)
    display_graph(G, pos,"Epistatic Network Graph", outpath, datefilter)
    print("=====================> All Processes are completed!")
 

if __name__ == "__main__":

    doc = """ Fitness vector generator
    for preprocessing  1.  usage python fitness_vector.py -p -in usa_matrix.csv -d usa_dates.txt -o .
    for fitness vector 2.  usage python fitness_vector.py -in complete_USA_sorted_with_dates.pkl -df 2020-04-06 -pv 0.5 -sn 5 -o . 
    for fitness vector 3.  usage python fitness_vector.py -in complete_USA_sorted_with_dates.pkl -df 2020-04-06 -fd 2020-01-17 -th 0.05 -sn 5 -ct ratio -o output/

    -fd, --start-date       Date after which sequence is needed the given date is not included, the format is yyyy-mm-dd. e.g. "2020-04-06"
    -df, --date-filter      Date till which sequence is needed, the format is yyyy-mm-dd. e.g. "2020-04-06"
    -th, --p-value          default=0.05    P value threshold for calculating edge significance.
    -sn, --shuffle-times    default=5       Number of times shuffling is to be done not considered with binomial calc_type.
    -ct, --calculation-type default="ratio" Type of calculation for SNP pairs possible values (ratio, diff, binomial).
    -sf  --save-files       default=False   Save calculated and processed data in files.
    """

    parser = argparse.ArgumentParser(description = doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-p', '--preprocess', action="store_true", help='Enable preprocessing')
    parser.add_argument('-o', '--output-path', type=str, default='.', help='output path')
    parser.add_argument('-d', '--date-path', type=str, help='Path for the date file.')
    parser.add_argument('-in', '--input', type=str, help='Path to the input file csv or the pkl file.')

    if not parser.parse_known_args()[0].preprocess: # check prepross flag
        parser.add_argument('-fd', '--start-date', type=str, default="2020-01-17", help='Date after which sequence is needed the given date is included, the format is yyyy-mm-dd. e.g. "2020-04-06"')
        parser.add_argument('-df', '--date-filter', type=str, help='Date till which sequence is needed, the format is yyyy-mm-dd. e.g. "2020-04-06"')
        parser.add_argument('-th', '--p-value', type=float, default=0.05, help='P value threshold for calculating edge significance')
        parser.add_argument('-sn', '--shuffle-times', type=int, default=5, help='Number of times shuffling is to be done not considered with binomial calc_type.')
        parser.add_argument('-ct', '--calculation-type', type=str, default="ratio", help='Type of calculation for SNP pairs possible values (ratio, diff, binomial).')
        parser.add_argument('-sf', '--save-files', type=bool, default=False, help='Save calculated and processed data in files.')


    args = parser.parse_args()


    if args.preprocess:
        if not args.date_path or not args.input:
            parser.error("--date-path and --input are required for preprocessing")
        else:
            preprocess_data(input = args.input, 
                    datepath = args.date_path,
                    outpath = args.output_path)

    else:
        if not args.input or not args.date_filter:
            parser.error("--input and date-filter are required")
        else:
            if os.path.exists(os.path.join(args.output_path, args.date_filter)):
                shutil.rmtree(os.path.join(args.output_path, args.date_filter))
                os.makedirs(os.path.join(args.output_path, args.date_filter))
            else:
                os.makedirs(os.path.join(args.output_path, args.date_filter))

            print(f"Directory created at {os.path.join(args.output_path, args.date_filter)}")

            with open (os.path.join(args.output_path, args.date_filter, f"statistics_{args.date_filter}.log"),"w") as logfile:
                logfile.write(f"Running with the following parameters: \n"
                              f"    1. Input file path: {args.input}\n"
                              f"    2. Date range: {args.start_date} - {args.date_filter}\n"
                              f"    3. P value threshold for significant edges:  {args.p_value} {'(not considered)' if args.calculation_type=='binomial' else ''}\n"
                              f"    4. No of shuffles :  {args.shuffle_times} {'(not considered)' if args.calculation_type=='binomial' else ''}\n"
                              f"    5. Output path:  {args.output_path} \n"
                              f"    5. Calculation type:  {args.calculation_type} \n"
                                )
                
            main(input = args.input, 
                start_date = args.start_date,
                datefilter = args.date_filter,
                p_val = args.p_value,
                shuffles = args.shuffle_times,
                outpath = args.output_path,
                calc_type = args.calculation_type,
                save_files = args.save_files
                )

# window
# clustering with gap
# 