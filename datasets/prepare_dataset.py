import os
import shutil
import get_cell_density, get_entropy, get_Sorensons_neighborhood_similarity 

source = 'datasets/tiles1/'
folder = 'train/'
csv_path = source + folder
tmp1_path = source + 'tmp1/'
tmp2_path = source + 'tmp2/'
outPath = source + 'out/' + folder


# create tmp1, tmp2 and out folders
if not os.path.exists(tmp1_path):
    os.makedirs(tmp1_path)
if not os.path.exists(tmp2_path):
    os.makedirs(tmp2_path)
if not os.path.exists(outPath):
    os.makedirs(outPath)

# get file names in the path
files = os.listdir(csv_path)
# remove the file extension from the file names and _edges, _nodes
file_names = [file.split('.csv')[0] for file in files if file.endswith('.csv')]
file_names = [file.split('_edges')[0] for file in file_names]
file_names = [file.split('_nodes')[0] for file in file_names]
# remove duplicates
file_names = list(set(file_names))

get_cell_density.get_cell_density(file_names, csv_path, tmp1_path)
get_entropy.get_entropy(file_names, tmp1_path, tmp2_path)
get_Sorensons_neighborhood_similarity.get_sorenson_similarity(file_names, tmp2_path, outPath)

# remove tmp1 and tmp2 folders
if os.path.exists(tmp1_path):
    shutil.rmtree(tmp1_path)
if os.path.exists(tmp2_path):
    shutil.rmtree(tmp2_path)

