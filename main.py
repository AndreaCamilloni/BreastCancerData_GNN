import json
import os
import random
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



from datasets import node_prediction
from models import EGNNC, MLPTwoLayers, CombinedModel, gCNN
import utils

random.seed(0)

def main():
    config = utils.parse_args()
    device = 'cuda:0' if config['cuda'] and torch.cuda.is_available() else 'cpu'
    config['device'] = device
    graphlet_size = config['graphlet_size'] 
    training = True if not config['val'] and not config['test'] else False

    mlp_out_dim = 1 # Tumor Classification
    dataset_args = ('train', config['num_layers']) if not config['val'] and not config['test'] else ('val', config['num_layers']) if config['val'] else ('test', config['num_layers'])
    datasets = utils.get_node_dataset_gcn(dataset_args, config['dataset_folder'], is_debug=config["is_debug"], num_classes = mlp_out_dim)
    loaders = [DataLoader(dataset=ds, batch_size=config['batch_size'], shuffle=True, collate_fn=ds.collate_wrapper) for ds in datasets]

    input_dim, hidden_dim, output_dim = datasets[0].get_dims()[0], config['hidden_dims'][0], config['out_dim']
    channel_dim = datasets[0].get_channel()

    
    model = EGNNC(input_dim, hidden_dim, output_dim, channel_dim, config['num_layers'], config['dropout'], config['device']).to(config['device'])

    if config['classifier'] == 'mlp':
        classifier = MLPTwoLayers(input_dim=channel_dim*output_dim, hidden_dim=output_dim, output_dim=mlp_out_dim, dropout=0.5).to(config["device"])
    elif config['classifier'] == 'gCNN':
        #graphlet_size = 10
        classifier = gCNN(num_channels = graphlet_size+1, output_dim = 1, dropout=0.5, device=config["device"], input_size = 16, training = training)
    
    combined_model = CombinedModel(model, classifier, config['classifier'], config["device"]).to(config["device"])

    print("Combined model: ", combined_model)

    if config['val'] or config['test']:
        path = os.path.join(config['saved_models_dir'], utils.get_fname(config))
        combined_model.load_state_dict(torch.load(path, map_location=torch.device(device)))

    criterion = utils.compute_loss
    optimizer = optim.Adam(combined_model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.5)

    if not config['val'] and not config['test']:
        combined_model.train()
        for epoch in range(config['epochs']):
            #print(f"Epoch {epoch}")
            for i, loader in enumerate(loaders):
                
                for idx, batch in enumerate(loader):
                    adj, features, edge_features, nodes, labels, _ = batch
                    adj, labels, features, edge_features = adj.to(device), labels.to(device), features.to(device), edge_features.to(device)
                    scores = combined_model(features, edge_features, nodes, graphlet_size, adj)
                    #print("scores: ", scores)
                    #print("labels: ", labels)
                    loss = criterion(scores, labels, num_classes=mlp_out_dim)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            
            print(f"Epoch {epoch} loss: {loss.item()}")

            scheduler.step()

        # Save trained model
        if config['save']:
            path = os.path.join(config['saved_models_dir'], utils.get_fname(config))
            torch.save(combined_model.state_dict(), path)
            print(f"Saved model to {path}")

        # save results of node classification
        combined_model.eval()
        #epoch = 0
        for i, loader in enumerate(loaders):
            #print("epoch: ", epoch + 1  )
            all_node_ids = []
            all_true_labels = []
            all_scores = []

            for idx, batch in enumerate(loader):
                adj, features, edge_features, nodes, labels, _ = batch
                labels, features, edge_features = labels.to(device), features.to(device), edge_features.to(device)
                scores = combined_model(features, edge_features, nodes, graphlet_size, adj)
                
                # Convert to CPU and NumPy for saving
                labels = labels.cpu().numpy()
                scores = scores.detach().cpu().numpy()
                
                # Append to lists
                all_node_ids.extend(nodes.tolist())
                all_true_labels.extend(labels.tolist())
                all_scores.extend(scores.tolist())
                
            # Sort the accumulated results by node_id
            sorted_results = sorted(zip(all_node_ids, all_true_labels, all_scores), key=lambda x: x[0])
            
            # Unzip back into separate lists
            sorted_node_ids, sorted_true_labels, sorted_scores = zip(*sorted_results)
            
            
            #filtered_true_labels = [label for label in sorted_true_labels if label!=-1]
            #filtered_scores = [score for label, score in zip(sorted_true_labels, sorted_scores) if label!=-1]

            # Save sorted results
            image_name = datasets[i].path[0].split("\\")[1]
            name = f'results_{image_name}_nodes.json'
            task = 'train'
            filename = os.path.join(config['results_dir'],'node_classification_results', task,  name)
            with open(filename, 'w') as f:
                json.dump({
                    'sorted_node_ids': sorted_node_ids,
                    'sorted_true_labels': sorted_true_labels,
                    'sorted_scores': sorted_scores
                }, f)

    # Evaluation (Validation or Test)
    if config['val'] or config['test']:
        combined_model.eval()

        all_true_values = []
        all_predicted_values = []

        for i, loader in enumerate(loaders):
            print(i / len(loaders))
            all_node_ids = []
            all_true_labels = []
            all_scores = []
            
            for idx, batch in enumerate(loader):
                adj, features, edge_features, nodes, labels, _ = batch
                labels, features, edge_features = labels.to(device), features.to(device), edge_features.to(device)

                # Graphsage needs the edges as adjacency list
                #edge_index = adj[0].nonzero().T

                scores = combined_model(features, edge_features, nodes, graphlet_size, adj)
                
                # Convert to CPU and NumPy for saving
                labels = labels.cpu().numpy()
                scores = scores.detach().cpu().numpy()
                
                # Append to lists
                all_node_ids.extend(nodes.tolist())
                all_true_labels.extend(labels.tolist())
                all_scores.extend(scores.tolist())
                
            # Sort the accumulated results by node_id
            sorted_results = sorted(zip(all_node_ids, all_true_labels, all_scores), key=lambda x: x[0])
            
            # Unzip back into separate lists
            sorted_node_ids, sorted_true_labels, sorted_scores = zip(*sorted_results)

            all_true_values += list(sorted_true_labels)
            all_predicted_values += list(sorted_scores)

            # Save sorted results
            # image_name = datasets[i].path[0].split("\\")[1]
            # name = f'results_{image_name}_nodes.json'
            # task = 'val' if config['val'] else 'test'
            # filename = os.path.join(config['results_dir'],'node_classification_results', task,  name)
            # with open(filename, 'w') as f:
            #     json.dump({
            #         'sorted_node_ids': sorted_node_ids,
            #         'sorted_true_labels': sorted_true_labels,
            #         'sorted_scores': sorted_scores
            #     }, f)
        
        accuracy = accuracy_score(all_true_values, np.round(all_predicted_values))
        precision = precision_score(all_true_values, np.round(all_predicted_values))
        recall = recall_score(all_true_values, np.round(all_predicted_values))
        f1 = f1_score(all_true_values, np.round(all_predicted_values))
        #dice = utils.dice_score(filtered_true_labels, np.round(filtered_scores))
        

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")

        

if __name__ == '__main__':
    main()







    