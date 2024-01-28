
import csv
import os
from tempfile import tempdir
from tkinter import E
import pandas as pd
import math

 
def get_entropy(file_names, csv_path, outPath):
     
    for file in file_names:
        edge_csv_file = file.strip()+'_edges.csv'
        node_csv_file = file.strip() + '_nodes.csv'
        adj_list = {}
        with open (csv_path + edge_csv_file) as edges:
            print(file)
            edge_reader = csv.reader(edges, delimiter =',')

            node_cell_type_count = {}
            for eachEdge in edge_reader:
                source , target = eachEdge[0], eachEdge[1]

                node_reader = csv.reader(open(csv_path+node_csv_file, 'r', encoding='iso-8859-1'))
                if source != 'source':    
                    for eachRow in node_reader:
                        if source == eachRow[0]: 
                            source_type = eachRow[3] #get cell-type of the source node
                        if target == eachRow[0]:
                            target_type = eachRow[3] #get cell-type of the target node   

                    ### Add neighbours of source              
                    if source in node_cell_type_count.keys():
                        temp_source = node_cell_type_count[source]
                    else: 
                        #{0: 'AMBIGUOUS', 1: 'nonTIL_stromal', 2: 'other', 3: 'sTIL', 4: 'tumor'}
                        temp_source = {'AMBIGUOUS': 0, 'nonTIL_stromal':0, 'other': 0, 'sTIL':0,'tumor':0, 'total': 0}

                    for eachKey in temp_source.keys():
            
                        if target_type == eachKey:
                            temp_source[target_type] = int(temp_source[target_type]) + 1 # add the count of the node_type
                            temp_source['total'] = int(temp_source['total']) + 1 # add the total count of all neighbours
                    

                    node_cell_type_count[source] = temp_source

                    ### Add neighbours of target
                    if target in node_cell_type_count.keys():
                        temp = node_cell_type_count[target]
                    else: 
                        temp = {'AMBIGUOUS': 0, 'nonTIL_stromal':0, 'other': 0, 'sTIL':0,'tumor':0, 'total': 0}

                    for eachKey in temp.keys(): 
                        if source_type == eachKey: 
                            temp[source_type] = int(temp[source_type]) + 1
                            temp['total'] = int(temp['total']) + 1
                

                    node_cell_type_count[target] = temp

                    #print(node_cell_type_count)
                        
            
            node_reader = csv.reader(open(csv_path+node_csv_file, 'r', encoding='iso-8859-1'))
            writer = csv.writer(open(outPath+node_csv_file, 'w', newline=''))   
            node_entropy_dict = {}
            for eachRow in node_reader: 
                eachRow = eachRow
                node_entropy = 'Node_entropy'
                id = eachRow[0]
             
                if id in node_cell_type_count.keys(): 
                    
                    id_type = eachRow[3]
                    entropy_list = node_cell_type_count[id]
                    #print(entropy_list)
                    ###Add self node type for entropy calculation
                    entropy_list[id_type] = int(entropy_list[id_type]) +1
                    entropy_list['total'] = int(entropy_list['total']) +1

                    ###Calculate node type probabilities
                    prob_AMB = float(entropy_list['AMBIGUOUS'])/ float(entropy_list['total'])
                    prob_nonTIL_stroma = float(entropy_list['nonTIL_stromal'])/ float(entropy_list['total'])
                    prob_oth = float(entropy_list['other'])/ float(entropy_list['total'])
                    prob_sTIL = float(entropy_list['sTIL'])/ float(entropy_list['total'])
                    prob_tum = float(entropy_list['tumor'])/float(entropy_list['total'])

                    if prob_AMB == 0.0: 
                        entropy_AMB = 0
                    else: 
                        entropy_AMB = abs(prob_AMB * math.log(prob_AMB)) 

                    if prob_nonTIL_stroma == 0.0: 
                        entropy_nonTIL_stroma = 0
                    else: 
                        entropy_nonTIL_stroma = abs(prob_nonTIL_stroma * math.log(prob_nonTIL_stroma)) 

                    if prob_oth == 0.0: 
                        entropy_oth = 0
                    else: 
                        entropy_oth = abs(prob_oth * math.log(prob_oth)) 

                    if prob_sTIL == 0.0: 
                        entropy_sTIL = 0
                    else: 
                        entropy_sTIL = abs(prob_sTIL * math.log(prob_sTIL)) 

                    if prob_tum == 0.0:
                        entropy_tum = 0.0
                    else: 
                        entropy_tum = abs(prob_tum * math.log(prob_tum)) 


                    node_entropy = entropy_AMB + entropy_oth + entropy_sTIL + entropy_nonTIL_stroma + entropy_tum
                    node_entropy_dict[id] = node_entropy
                
                else:
                   node_entropy = 'Node_Entropy' 
                
                eachRow.append(node_entropy)
                writer.writerow(eachRow)
        edges.close()

        with open (csv_path + edge_csv_file) as edges:
            edge_reader = csv.reader(edges, delimiter =',')
            edge_writer = csv.writer(open(outPath+edge_csv_file, 'w',newline=''))
            delta_entropy = 'Delta_Entropy'
            for eachEdge in edge_reader:
                source , target = eachEdge[0], eachEdge[1]
                if source != 'source' and target != 'target' and node_entropy_dict[source] != 'Node_Entropy' and node_entropy_dict[target] != 'Node_Entropy':
                    delta_entropy = abs(float(node_entropy_dict[source]) - float(node_entropy_dict[target]))
                else: 
                    delta_entropy = 'Delta_Entropy'
                eachEdge.append(delta_entropy)
                edge_writer.writerow(eachEdge)
        edges.close()

 