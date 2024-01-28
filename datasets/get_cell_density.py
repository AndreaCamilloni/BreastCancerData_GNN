import csv
import os
import pandas as pd

 
def get_cell_density(file_names, csv_path, outPath):
    #graphs = graphList.readlines()
    for file in file_names:
        edge_csv_file = file.strip()+'_edges.csv'
        node_csv_file = file.strip() + '_nodes.csv'

        with open (csv_path + edge_csv_file) as edges:
            edge_reader = csv.reader(edges, delimiter =',')
            node_cell_density = {}
            node_cell_count = {}
            node_cell_distance = {}   
            node_type_density = {}
            
            edge_writer = csv.writer(open(outPath+edge_csv_file, 'w', newline=''))
            for eachEdge in edge_reader:
                edge_writer.writerow(eachEdge) # copy the edges to the new folder
            
                source , target = eachEdge[0], eachEdge[1]
                dist = eachEdge[2]
                node_reader = csv.reader(open(csv_path+node_csv_file, 'r'))
                if source != 'source':    
                    for eachRow in node_reader:
                        if source == eachRow[0]: 
                            source_type = eachRow[4] #get cell-type of the source node
                        if target == eachRow[0]:
                            target_type = eachRow[4] #get cell-type of the target node                
                    if source in node_cell_count.keys():
                        node_cell_count[source] += 1
                        node_cell_distance[source] = node_cell_distance[source] + float(dist)
                    else: 
                        node_cell_count[source] = 1
                        node_cell_distance[source] = float(dist)

                    if target in node_cell_count.keys():
                        node_cell_count[target] += 1
                        node_cell_distance[target] = node_cell_distance[target] + float(dist)

                    else: 
                        node_cell_count[target] = 1  
                        node_cell_distance[target] = float(dist)  

            for eachKey in node_cell_distance.keys():
                node_cell_density[eachKey] = node_cell_distance[eachKey]/ node_cell_count[eachKey]
            
            node_reader = csv.reader(open(csv_path+node_csv_file, 'r'))
            writer = csv.writer(open(outPath+node_csv_file, 'w', newline=''))   
            for eachRow in node_reader: 
                id = eachRow[0]
                if id in node_cell_density.keys():
                    cell_density = node_cell_density[id]
                else:
                    if id !='id':
                        cell_density = 0.0 
                    else:
                        cell_density = 'Cell_density'

                eachRow.append(cell_density)
                writer.writerow(eachRow)

                 
        edges.close()

 