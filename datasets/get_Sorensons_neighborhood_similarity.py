
import csv
import os
from tempfile import tempdir
from tkinter import E
import pandas as pd
import math


 
def get_sorenson_similarity(file_names, csv_path, outPath):
    
    for eachGraph in file_names:
        edge_csv_file = eachGraph.strip()+'_edges.csv'
        adj_list = {}

        with open (csv_path + edge_csv_file) as edges:
            edge_reader = csv.reader(edges, delimiter =',')
            for eachEdge in edge_reader:
                source , target = eachEdge[0], eachEdge[1]
                if source != 'source' and target != 'target':
                    if source in adj_list.keys():
                        temp = []
                        temp = list(adj_list[source])
                        temp.append(target)
                        adj_list[source] = temp
                    else: 
                        adj_list[source] = target

                    if target in adj_list.keys():
                        temp = []
                        temp = list(adj_list[target])
                        temp.append(source)
                        adj_list[target] = temp
                    else: 
                        adj_list [target] = source
        edges.close()
        
        with open (csv_path + edge_csv_file) as edgeFile:
            edge_reader = csv.reader(edgeFile, delimiter =',')
            writer = csv.writer(open(outPath+edge_csv_file, 'w', newline='')) 
            for eachEdge in edge_reader:
                #print(eachEdge)
                source , target = eachEdge[0], eachEdge[1]
                if source != 'source' and target != 'target':
                    if source in adj_list.keys():
                        source_list = list(adj_list[source])
                        target_list = list(adj_list[target])

                        source_set = set(source_list)
                        common_elements = source_set.intersection(target_list)
                        common_elements = list(common_elements)
                        sorenson_similarity = float(2 * len(common_elements) / (len(source_list) + len(target_list)))
                        #print(sorenson_similarity)
                    else: 
                        sorenson_similarity = 0.0
                else: 
                    sorenson_similarity = 'Sorenson_Similarity'
                
                #print(edge_csv_file)

                eachEdge.append(sorenson_similarity)
                writer.writerow(eachEdge)

        edgeFile.close()
        # copy the nodes file to the output folder
        nodes_csv_file = eachGraph.strip()+'_nodes.csv'
        df = pd.read_csv(csv_path + nodes_csv_file)
        df.to_csv(outPath + nodes_csv_file, index=False)


 

            
        