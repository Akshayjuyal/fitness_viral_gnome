"""
Author: Akshay Juyal (Department of Computer Science GSU)
Date: March 19, 2024
Description: This script performs preprocessing of covid amino acid sequence data or
            similar RNA based viral sequences as well as 
            calculate the fitness value for each sequences
"""
import os
import pickle
import argparse
import shutil
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import networkx as nx
from scipy.stats import binom
from scipy.special import comb


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
    with Pool(processes=30) as pool:  
        results = list(tqdm(pool.imap(calculate_pair_statistics, pair), total=len(pair), desc="Computing Pair ratios"))
    return results


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
    
def create_adj_matrix_binom(columns_count,original_p):
    # [0, int(idx), int(in_idx), float((((z_z * o_o)+1)) / (((z_o * o_z)+1))), z_z, z_o, o_z, o_o]
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
    _, columns_count = original_df.shape

    if calc_type == "binomial":
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
    norm_fitness_vector = [fitness_val - min_fitness if fitness_val!=0 else fitness_val for fitness_val in fitness_vector] # this is what you need !!!
    print("=====================> Saving Normalized Fitness vector")
    np.savetxt(os.path.join(outpath, datefilter, f"norm_fitnessVector_{datefilter}.csv"), norm_fitness_vector, fmt="%d", delimiter=",")

 

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
            pass
            # preprocess_data(input = args.input, 
            #         datepath = args.date_path,
            #         outpath = args.output_path)

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