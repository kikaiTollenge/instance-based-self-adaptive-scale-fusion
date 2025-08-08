import sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.abmil import GatedAttention
from model.transmil import TransMIL
from model.transformer import ACMIL_GA
from model.clam import CLAM_MB,CLAM_SB
import os
from dataset.dataset import pkl_Dataset
from model.adaptiveScaleFusion import decouple_feature
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import datetime
import argparse
import csv
from sklearn.metrics import roc_auc_score, recall_score, precision_score
import torch.nn.functional as F
def get_weight_bins(weights):
    bins = torch.linspace(0, 1, 11).to('cuda:0')  
    weights = weights.contiguous()
    bin_indices = torch.bucketize(weights, bins) - 1  
    bin_indices = torch.clamp(bin_indices, min=0, max=9)  
    return bin_indices

def record_and_plot_weight_bins(model, dataloader, weight_path,args,i):
    model.load_state_dict(torch.load(weight_path))
    model.to('cuda:0')
    high_bin_counts = torch.zeros(10, dtype=torch.int).to('cuda:0')
    low_bin_counts = torch.zeros(10, dtype=torch.int).to('cuda:0')
    model.eval()

    for feature,label in dataloader:
        x = feature.to('cuda:0')

        with torch.no_grad():
            x = x.squeeze(0)
            high_feature, low_feature = decouple_feature(x)

            high_K = model.multi_weights.high_Q(high_feature)
            low_K = model.multi_weights.low_Q(low_feature)
            
            high_sim = F.cosine_similarity(high_feature, high_K, dim=1).unsqueeze(1)
            low_sim = F.cosine_similarity(low_feature, low_K, dim=1).unsqueeze(1)

            weights = torch.hstack([high_sim, low_sim])
            weights = F.softmax(weights, dim=1)

            high_weights_softmax = weights[:,0].unsqueeze(0).permute(1,0)
            low_weights_softmax = weights[:,1].unsqueeze(0).permute(1,0)

            high_bin_indices = get_weight_bins(high_weights_softmax)
            low_bin_indices = get_weight_bins(low_weights_softmax)


            for idx in high_bin_indices:
                high_bin_counts[idx] += 1
            for idx in low_bin_indices:
                low_bin_counts[idx] += 1


    bins = ['[0.0, 0.1)', '[0.1, 0.2)', '[0.2, 0.3)', '[0.3, 0.4)', '[0.4, 0.5)', 
            '[0.5, 0.6)', '[0.6, 0.7)', '[0.7, 0.8)', '[0.8, 0.9)', '[0.9, 1.0)']
    
    plt.figure(figsize=(12, 6))


    plt.bar(bins, high_bin_counts.cpu().numpy())
    plt.title("High Feature Weights Distribution")
    plt.xlabel("Weight Bins")
    plt.ylabel("Number of Instances")

    plt.tight_layout()
    plt.show()
    plt.savefig(f'feature_weights_distribution.jpg')
    print("High Feature Weights Distribution SAVED")

class EarlyStopping:
    def __init__(self, patience=5, stop_epoch=15, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train(dataloader, model, loss_fn, optimizer, device, epoch, multi_scale, args):
    model.train()
    running_loss = 0.0
    for feature, label in dataloader:
        optimizer.zero_grad()
        model.to(device)
        feature = feature.to(device)
        label = label.to(device)
        if args.model == 'clam_sb' or args.model == 'clam_mb': 
            output,instance_loss = model(feature, label=label,instance_eval = True,multi_scale=multi_scale)
            loss = loss_fn(output, label)
            loss = loss + instance_loss

        elif args.model == 'acmil':
            sub_preds, output, attn = model(feature,multi_scale = multi_scale)
            if args.n_tokens > 1:
                loss0 = loss_fn(sub_preds, label.repeat_interleave(args.n_tokens))
            else:
                loss0 = torch.tensor(0,)
            loss1 = loss_fn(output,label)
            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            attn = torch.softmax(attn, dim=-1)

            for i in range(args.n_tokens):
                for j in range(i + 1, args.n_tokens):
                    diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                                args.n_tokens * (args.n_tokens - 1) / 2)
            
            loss = diff_loss + loss0 + loss1
        else:
            output = model(feature, multi_scale=multi_scale)
            loss = loss_fn(output, label)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    total_loss = running_loss / len(dataloader)
    print("Epoch: {}, Loss: {}".format(epoch+1, total_loss))
    return total_loss


def eval(dataloader, model, loss_fn, device, epoch, multi_scale, args):
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    all_probs = []

    false_positives = 0
    false_negatives = 0
    total_0 = 0
    total_1 = 0

    with torch.no_grad():
        for feature, label in dataloader:
            feature = feature.to(device)
            label = label.to(device)
            model.to(device)
            if args.model == 'clam_sb' or args.model == 'clam_mb': 
                output,instance_loss = model(feature, label=label,instance_eval = True,multi_scale=multi_scale)
                extra_loss = instance_loss
                loss = loss_fn(output, label)
                loss = loss + extra_loss

            elif args.model == 'acmil':
                sub_preds, output, attn = model(feature, multi_scale=multi_scale)
                if args.n_tokens > 1:
                    loss0 = loss_fn(sub_preds, label.repeat_interleave(args.n_tokens))
                else:
                    loss0 = torch.tensor(0,)
                loss1 = loss_fn(output,label)
                diff_loss = torch.tensor(0).to(device, dtype=torch.float)
                attn = torch.softmax(attn, dim=-1)

                for i in range(args.n_tokens):
                    for j in range(i + 1, args.n_tokens):
                        diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                                    args.n_tokens * (args.n_tokens - 1) / 2)
                
                loss = diff_loss + loss0 + loss1
            else:
                output = model(feature, multi_scale=multi_scale)
                loss = loss_fn(output, label)

            val_loss += loss.item()

            probs = torch.softmax(output, dim=1)[:, 1]
            _, predicted = torch.max(output, 1)

            all_labels.extend(label.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0)
            total_0 += (label == 0).sum().item()
            total_1 += (label == 1).sum().item()
            false_positives += ((predicted == 1) & (label == 0)).sum().item()
            false_negatives += ((predicted == 0) & (label == 1)).sum().item()

    val_loss /= len(dataloader)
    accuracy = total_correct / total_samples
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else 0.0
    recall = recall_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)

    print(f"Eval Epoch {epoch+1} - Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, AUC: {auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    return val_loss, accuracy, auc, recall, precision, false_positives, false_negatives, total_0, total_1


def plot_curves(train_losses, val_losses, accuracies, save_path="training_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_experiment_folder(base_path='./result', subfolder='c16_single_low',source='ctrans',concat_type='none',model_name='transmil',alpha = None):
    if concat_type == 'none':
        if alpha == None:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_path = os.path.join(base_path, subfolder+'_'+source+'_'+model_name, current_time)
        else:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_path = os.path.join(base_path, subfolder+'_'+source+'_'+model_name+'_alpha_'+str(alpha), current_time)
        os.makedirs(save_path, exist_ok=True)
    elif concat_type == 'concat':
        if alpha is None:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_path = os.path.join(base_path, subfolder+'_'+source+'_'+model_name+'_new_model', current_time)
        else:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_path = os.path.join(base_path, subfolder+'_'+source+'_'+model_name+'_alpha_'+str(alpha)+'_new_model', current_time)
        os.makedirs(save_path, exist_ok=True)
    return save_path


def save_best_results_to_csv(best_results, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'Retry', 'Best Val Loss', 'Best Accuracy',
            'Best AUC', 'Best Recall', 'Best Precision',
            'False Positives', 'False Negatives', 'Total 0', 'Total 1'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in best_results:
            writer.writerow(result)
            
def build_model(model_name: str, model_config: dict):
    if model_name == 'abmil':
        return GatedAttention(config=model_config)
    elif model_name == 'transmil':
        return TransMIL(model_config)
    elif model_name == 'acmil':
        return ACMIL_GA(conf=model_config)
    elif model_name == 'clam_sb':
        return CLAM_SB(model_config, gate=True, size_arg="small",
                       k_sample=8, dropout=True)
    elif model_name == 'clam_mb':
        return CLAM_MB(model_config, gate=True, size_arg="small",
                       k_sample=8, dropout=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MIL Model')
    parser.add_argument('--config', type=str, default='c16_single_low', help='dataset config')
    parser.add_argument('--alpha', type=float, default=2.0, help='fusion weight')
    parser.add_argument('--source', type=str, default='moco', help='feature extractor different source')
    parser.add_argument('--concat', type=str, default='none', help='concat type must be concat or none')
    parser.add_argument('--model', type=str, default='abmil', help='model type')
    parser.add_argument('--n_tokens', type = int, default=5, help='parameter for ACMIL')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    args = parser.parse_args()
    # set_seed(42)
    train_config = {
        'c16_single_low': {'dataset': 'c16', 'model_type': 'custom', 'alpha': 2.0, 'data_type': 'low'},
        'c16_single_high': {'dataset': 'c16', 'model_type': 'custom', 'alpha': 2.0, 'data_type': 'high'},
        'c16_multi_custom': {'dataset': 'c16', 'model_type': 'custom', 'alpha': args.alpha, 'data_type': 'both'},
        'c16_multi_weighted': {'dataset': 'c16', 'model_type': 'weighted', 'alpha': 0.5, 'data_type': 'both'},
        'tcga_single_low': {'dataset': 'tcga', 'model_type': 'custom', 'alpha': 2.0, 'data_type': 'low'},
        'tcga_single_high': {'dataset': 'tcga', 'model_type': 'custom', 'alpha': 2.0, 'data_type': 'high'},
        'tcga_multi_custom': {'dataset': 'tcga', 'model_type': 'custom', 'alpha': args.alpha, 'data_type': 'both'},
        'tcga_multi_weighted': {'dataset': 'tcga', 'model_type': 'weighted', 'alpha': 0.5, 'data_type': 'both'}
    }
    input_size ={
        'ctrans':768,
    }

    config = train_config[args.config]
    dataset = config['dataset']
    fusion_type = config['model_type']
    alpha = config['alpha']
    data_type = config['data_type']
    multi_scale = alpha > 0.0 and alpha < 1.0
    path_alpha = None if alpha == 2.0 else alpha
    input_size = input_size[args.source]
    concat_type = args.concat
    print(args.config,' ',args.source,' ',args.concat,' ',args.model)
    model_config = {
        'fusion_type':fusion_type,
        'alpha':alpha,
        'concat_type':concat_type,
        'input_size':input_size,
        'inner_size':int(input_size / 2),
        'factor': 2 if concat_type == 'concat' else 1, 
        'n_tokens':args.n_tokens 
    }


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    retrys = 20
    best_results = []

    train_dataset = pkl_Dataset(dataset, 'train', data_type, args.source)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = pkl_Dataset(dataset, 'test', data_type, args.source)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    learning_rates = [1e-4]

    for lr in learning_rates:
        print(f"lr is {lr}")
        for retry in range(retrys):
            folder = create_experiment_folder(subfolder=args.config,source=args.source,concat_type=concat_type,model_name= args.model,alpha=path_alpha)

            model = build_model(args.model, model_config).to(device)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

            train_losses = []
            val_losses = []
            accuracies = []

            early_stopping = EarlyStopping(patience=3, stop_epoch=10, verbose=True)
            best_acc = 0.0
            best_metrics = {}

            for epoch in range(50):
                print(f"Epoch {epoch + 1}/50\n" + "-" * 10)
                train_loss = train(train_dataloader, model, loss_fn, optimizer, device, epoch, multi_scale, args)
                val_loss, accuracy, auc, recall, precision, fp, fn, t0, t1 = \
                    eval(test_dataloader, model, loss_fn, device, epoch, multi_scale, args)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                accuracies.append(accuracy)
                scheduler.step(val_loss)
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_metrics = {
                        'Retry': retry + 1,
                        'Best Val Loss': val_loss,
                        'Best Accuracy': accuracy,
                        'Best AUC': auc,
                        'Best Recall': recall,
                        'Best Precision': precision,
                        'False Positives': fp,
                        'False Negatives': fn,
                        'Total 0': t0,
                        'Total 1': t1
                    }
                    torch.save(model.state_dict(), os.path.join(folder, "best_model.pt"))

                early_stopping(epoch, val_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

            plot_curves(train_losses, val_losses, accuracies, save_path=os.path.join(folder, "training_curves.png"))
            best_results.append(best_metrics)

        save_best_results_to_csv(best_results, os.path.join(os.path.dirname(folder), f"{lr}_best_results.csv"))
        print("All best results saved.")