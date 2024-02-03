import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score

class MentionPooling(torch.nn.Module):
    def __init__(self, pool_type="average"):
        super(MentionPooling, self).__init__()
        self.pooling_type = pool_type

    def forward(self, sequence_embeddings, special_tokens_mask, extract_mention = False):

        mask_indices = special_tokens_mask.nonzero(as_tuple=True)[-1]
        
        # Reshape the mask indices into pairs of [Ms] and [Me]
        mask_indices_pairs = mask_indices.view(-1, 2)

        pooled_representations = []

        for (ms_index, me_index), batch_embedding in zip(mask_indices_pairs, sequence_embeddings):
            # Extract specified embeddings
            if extract_mention == True:
                indexed_embeddings = batch_embedding[ms_index + 1:me_index, :]
            else: 
                indexed_embeddings = batch_embedding[[ms_index, me_index], :]

            if self.pooling_type == 'concatenation':
                pooled_representation = indexed_embeddings.unsqueeze(0)
            elif self.pooling_type == 'average':
                pooled_representation = torch.mean(indexed_embeddings, dim=0, keepdim=True)
            elif self.pooling_type == 'max':
                pooled_representation, _ = torch.max(indexed_embeddings, dim=0, keepdim=True)
            else:
                raise ValueError("Invalid pooling_type. Choose from 'concatenation', 'average', 'max', etc.")
            
            
            pooled_representations.append(pooled_representation)
    
        final_representation = torch.cat(pooled_representations, dim=0)
        
        return final_representation
    

class HierarchyClassifier:
    def __init__(self, model, optimizer, parent_classifier, child_classifier, optimizer_parent, optimizer_child, criterion):
        self.model = model
        self.optimizer = optimizer
        self.parent_classifier = parent_classifier
        self.child_classifier = child_classifier
        self.criterion = criterion

        self.optimizer_parent = optimizer_parent
        self.optimizer_child = optimizer_child

        self.losses =[]
        self.parent_losses = []
        self.child_losses = []
        self.accuracy_child = []
        self.accuracy_parent = []

        self.val_losses =[]
        self.val_parent_losses = []
        self.val_child_losses = []
        self.val_accuracy_child = []
        self.val_accuracy_parent = []

    def forward(self, input_ids, attention_mask, special_tokens_mask, pooling_function):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state

        # Apply pooling over special tokens using the provided pooling function
        pooled_output = pooling_function(last_hidden_states, special_tokens_mask)

        # Forward pass through classifiers
        parent_logits = self.parent_classifier(pooled_output)
        child_logits = self.child_classifier(pooled_output)

        return parent_logits, child_logits

    def validation(self, val_dataloader, device, pooling_function):

        total_loss = 0.0

        total_parent = 0.0
        total_child = 0.0
        total_parent_correct = 0.0
        total_child_correct = 0.0

        total_child_samples = 0
        total_parent_samples = 0

        # validation loop
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validation', leave=True):

                # Extract tensors from the batch dictionary
                input_ids = batch['input_ids'].to(device).squeeze(1)
                attention_mask = batch['attention_mask'].to(device).squeeze(1)
                token_type_ids = batch['token_type_ids'].to(device).squeeze(1)
                special_token_masks = batch['special_token_mask']
                parent_labels = batch['parent_label'].to(device)
                child_labels = batch['child_label'].to(device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                # Apply pooling over special tokens using the provided pooling function
                if pooling_function != None:
                    pooled_mentions = pooling_function(outputs.last_hidden_state, special_token_masks)
                    logits_child = self.child_classifier(pooled_mentions)
                    logits_parent = self.parent_classifier(outputs.pooler_output)
                else:
                    logits_parent = self.parent_classifier(outputs.pooler_output)
                    logits_child = self.child_classifier(outputs.pooler_output)

                # Calculate losses
                loss_parent = self.criterion(logits_parent, parent_labels)
                loss_child = self.criterion(logits_child, child_labels)

                loss = loss_parent + loss_child
                total_loss += loss.item()
                total_child += loss_child
                total_parent += loss_parent

                # Accuracy calculation
                _, predicted_parent = torch.max(logits_parent, 1)
                _, predicted_child = torch.max(logits_child, 1)

                total_parent_correct += (predicted_parent == parent_labels).sum().item()
                total_child_correct += (predicted_child == child_labels).sum().item()

                total_parent_samples += parent_labels.size(0)
                total_child_samples += child_labels.size(0)

        # Calculate average loss and accuracy
        average_loss = total_loss / len(val_dataloader)
        average_child = total_child/len(val_dataloader)
        average_parent = total_parent/len(val_dataloader)
        
        accuracy_parent = total_parent_correct / total_parent_samples
        accuracy_child = total_child_correct / total_child_samples

        self.val_losses.append(average_loss)
        self.val_child_losses.append(average_child)
        self.val_parent_losses.append(average_parent)

        self.val_accuracy_child.append(accuracy_child)
        self.val_accuracy_parent.append(accuracy_parent)

        # Print or store validation metrics
        print()
        print("Validation Metrcis")
        print(f'Val Avg Loss: {average_loss}, Val Parent Loss: {average_parent}, Val Child Loss: {average_child}')
        print(f'Val Parent Accuracy: {accuracy_parent}, Val Child Accuracy: {accuracy_child}')
        print()

    def train(self, train_dataloader, val_dataloader, num_epochs, foldername='model_checkpoints', paren_weight = 1, child_weight =1, train_BERT = True, train_child = True, train_parent = True, pooling_function=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)


        for epoch in range(num_epochs):
            total_loss = 0.0
            total_parent = 0.0
            total_child = 0.0
            total_parent_correct = 0.0
            total_child_correct = 0.0

            total_child_samples = 0
            total_parent_samples = 0

            self.model.train()

            for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True):

                if train_BERT == True:
                    self.optimizer.zero_grad()
                if train_parent == True:
                    self.optimizer_parent.zero_grad()
                if train_child == True:
                    self.optimizer_child.zero_grad()

                # Extract tensors from the batch dictionary
                input_ids = batch['input_ids'].squeeze(1).to(device).detach()
                attention_mask = batch['attention_mask'].squeeze(1).to(device)
                token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
                special_tokens_mask = batch['special_token_mask']
                parent_labels = batch['parent_label'].to(device)
                child_labels = batch['child_label'].to(device)

           
                # forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                # Apply pooling over special tokens using the provided pooling function
                if pooling_function != None:
                    pooled_mentions = pooling_function(outputs.last_hidden_state, special_tokens_mask)
                    logits_child = self.child_classifier(pooled_mentions)
                    logits_parent = self.parent_classifier(outputs.pooler_output)
                else:
                    logits_parent = self.parent_classifier(outputs.pooler_output)
                    logits_child = self.child_classifier(outputs.pooler_output)

                # calculate losses
                loss_parent = self.criterion(logits_parent, parent_labels)
                loss_child = self.criterion(logits_child, child_labels)

                loss = (loss_parent * paren_weight) + (loss_child * child_weight)
                total_loss += loss
                total_child += loss_child
                total_parent += loss_parent

                # backpropagation
                loss.backward()

                # update model parameters
                if train_BERT == False:
                    for param in self.model.parameters():
                        param.requires_grad = False
                else:
                    for param in self.model.parameters():
                        param.requires_grad = True
                    self.optimizer.step()

                if train_parent == False:
                    for param in self.parent_classifier.parameters():
                        param.requires_grad = False
                else:
                    for param in self.parent_classifier.parameters():
                        param.requires_grad = True
                    self.optimizer_parent.step()

                if train_child == False:
                    for param in self.child_classifier.parameters():
                        param.requires_grad = False
                else:
                    for param in self.child_classifier.parameters():
                        param.requires_grad = True
                    self.optimizer_child.step()


                # accuracy calculation
                _, predicted_parent = torch.max(logits_parent, 1)
                _, predicted_child = torch.max(logits_child, 1)

                total_parent_correct += (predicted_parent == parent_labels).sum().item()
                total_child_correct += (predicted_child == child_labels).sum().item()

                total_parent_samples += parent_labels.size(0)
                total_child_samples += child_labels.size(0)

            average_loss = total_loss/len(train_dataloader)
            average_child = total_child/len(train_dataloader)
            average_parent = total_parent/len(train_dataloader)

            # Calculate accuracy
            accuracy_parent = total_parent_correct / total_parent_samples
            accuracy_child = total_child_correct / total_child_samples

            # Create directory if it doesn't exist
            if not os.path.exists(foldername):
                os.makedirs(foldername)

            checkpoint_path = f'{foldername}/checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'parent_classifier_state_dict': self.parent_classifier.state_dict(),
                'optimizer_parent_state_dict': self.optimizer_parent.state_dict(),
                'child_classifier_state_dict': self.child_classifier.state_dict(),
                'optimizer_child_state_dict': self.optimizer_child.state_dict(),
                'loss': average_loss,
            }, checkpoint_path)

            self.losses.append(average_loss)
            self.child_losses.append(average_child)
            self.parent_losses.append(average_parent)
            self.accuracy_child.append(accuracy_child)
            self.accuracy_parent.append(accuracy_parent)

            # Print or store accuracy
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'Avg Loss: {average_loss}, Parent Loss: {average_parent}, Child Loss: {average_child}')
            print(f'Parent Accuracy: {accuracy_parent}, Child Accuracy: {accuracy_child}')

            self.validation(val_dataloader, device, pooling_function)

    def evaluate(self, dataloader, device, use_linkability_threshold=True, pooling_function=None):
        total_loss = 0.0
        total_parent = 0.0
        total_child = 0.0
        total_parent_correct = 0.0
        total_child_correct = 0.0
        linkable_child_correct = 0.0

        linkable_correct = 0.0
        linkable_total = 0.0

        total_child_samples = 0
        total_parent_samples = 0

        all_true_parent_labels = []
        all_pred_parent_labels = []
        all_true_child_labels = []
        all_pred_child_labels = []

        all_pred_linkable_labels = []

        # List to store true labels of unlinkable mentions
        unlinkable_true_labels = []

        # Evaluation loop
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluation', leave=True):
                # Extract tensors from the batch dictionary
                input_ids = batch['input_ids'].to(device).squeeze(1)
                attention_mask = batch['attention_mask'].to(device).squeeze(1)
                token_type_ids = batch['token_type_ids'].to(device).squeeze(1)
                special_tokens_mask = batch['special_token_mask']
                parent_labels = batch['parent_label'].to(device)
                child_labels = batch['child_label'].to(device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                if pooling_function != None:
                    last_hidden_state = outputs.last_hidden_state
                    pooled_mentions = pooling_function(last_hidden_state, special_tokens_mask)
                    logits_child = self.child_classifier(pooled_mentions)
                    logits_parent = self.parent_classifier(outputs.pooler_output)
                else:
                    logits_parent = self.parent_classifier(outputs.pooler_output)
                    logits_child = self.child_classifier(outputs.pooler_output)

                # Calculate losses
                loss_parent = self.criterion(logits_parent, parent_labels)
                loss_child = self.criterion(logits_child, child_labels)

                loss = loss_parent + loss_child
                total_loss += loss.item()
                total_child += loss_child
                total_parent += loss_parent

                _, predicted_parent = torch.max(logits_parent, 1)
                _, predicted_child = torch.max(logits_child, 1)

                if use_linkability_threshold == True:                                
                    linkable_true_child_labels = []
                    linkable_pred_child_labels = []

                    probability_child = torch.softmax(logits_child, dim=1)[:, 1].cpu().numpy()
                    linkability_threshold = 0.5
                    for i in range(len(probability_child)):
                        if probability_child[i] < linkability_threshold:
                            predicted_child[i] = 365
                            unlinkable_true_labels.append(child_labels[i].item())

                    for i in range(len(predicted_child)):
                        if predicted_child[i] != 365:
                            linkable_correct += (predicted_child[i] == child_labels[i]).item()
                            linkable_total += 1
                            linkable_true_child_labels.append(child_labels[i].item())
                            linkable_pred_child_labels.append(predicted_child[i].item())                    
                            
            print("1", predicted_parent, parent_labels)
            print("2", linkable_pred_child_labels, linkable_true_child_labels)
            total_parent_correct += (predicted_parent == parent_labels).sum().item()
            total_child_correct += (predicted_child == child_labels).sum().item()

            if use_linkability_threshold == True:   
                linkable_child_correct += (np.array(linkable_pred_child_labels) == np.array(linkable_true_child_labels)).sum().item()

            total_parent_samples += parent_labels.size(0)
            total_child_samples += child_labels.size(0)

            # Store true and predicted labels for precision, recall, and F1 score
            all_true_parent_labels.extend(parent_labels.cpu().numpy())
            all_pred_parent_labels.extend(predicted_parent.cpu().numpy())
            all_true_child_labels.extend(child_labels.cpu().numpy())
            all_pred_child_labels.extend(predicted_child.cpu().numpy())
            all_pred_linkable_labels.extend(linkable_pred_child_labels)

        # Calculate average loss and accuracy
        average_loss = total_loss / len(dataloader)
        average_child = total_child / len(dataloader)
        average_parent = total_parent / len(dataloader)

        accuracy_parent = total_parent_correct / total_parent_samples
        accuracy_child = total_child_correct / total_child_samples
        if use_linkability_threshold == True:   
            accuracy_linkable = linkable_child_correct / linkable_total

        # Calculate precision, recall, and F1 score
        precision_parent = precision_score(all_true_parent_labels, all_pred_parent_labels, average='macro')
        recall_parent = recall_score(all_true_parent_labels, all_pred_parent_labels, average='macro')
        f1_parent = f1_score(all_true_parent_labels, all_pred_parent_labels, average='macro')

        precision_child = precision_score(all_true_child_labels, all_pred_child_labels, average='macro')
        recall_child = recall_score(all_true_child_labels, all_pred_child_labels, average='macro')
        f1_child = f1_score(all_true_child_labels, all_pred_child_labels, average='macro')

        precision_link = precision_score(linkable_true_child_labels, all_pred_linkable_labels, average='macro')
        recall_link = recall_score(linkable_true_child_labels, all_pred_linkable_labels, average='macro')
        f1_link = f1_score(linkable_true_child_labels, all_pred_linkable_labels, average='macro')

        # Print or store evaluation metrics
        print()
        print("Evaluation Metrics")
        print(f'Eval Avg Loss: {average_loss}, Eval Parent Loss: {average_parent}, Eval Child Loss: {average_child}')
        print(f'Eval Parent Accuracy: {accuracy_parent}, Eval Child Accuracy: {accuracy_child}')
        print(f'Eval Parent Precision: {precision_parent}, Eval Parent Recall: {recall_parent}, Eval Parent F1: {f1_parent}')
        print(f'Eval Child Precision: {precision_child}, Eval Child Recall: {recall_child}, Eval Child F1: {f1_child}')
        if use_linkability_threshold == True:
            print(f'Eval Linkable-Only Accuracy: {accuracy_linkable}, Eval Linkable-Only Precision: {precision_link}, Eval Linkable-Only Recall: {recall_link}, Eval Linkable-Only F1: {f1_link}')
        print()

        return average_loss, average_parent, average_child, accuracy_parent, accuracy_child, \
               precision_parent, recall_parent, f1_parent, precision_child, recall_child, f1_child, unlinkable_true_labels
    

    def predict(self, tokenized_text,special_tokens_mask=None, pooling_function=None):

        input_ids = tokenized_text['input_ids']
        attention_mask = tokenized_text['attention_mask']
        token_type_ids = tokenized_text['token_type_ids']

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids)

            if pooling_function != None:
                last_hidden_state = outputs.last_hidden_state
                # Convert special_tokens_mask to a PyTorch tensor
                special_tokens_mask = torch.tensor(special_tokens_mask)

                pooled_mentions = pooling_function(last_hidden_state, special_tokens_mask)
                # print("Pooled Mentions: ", pooled_mentions)
                logits_child = self.child_classifier(pooled_mentions)
                logits_parent = self.parent_classifier(outputs.pooler_output)
            else:
                logits_parent = self.parent_classifier(outputs.pooler_output)
                logits_child = self.child_classifier(outputs.pooler_output)

            _, predicted_parent = torch.max(logits_parent, 1)
            _, predicted_child = torch.max(logits_child, 1)

            # Apply linkability threshold
            probability_child = torch.softmax(logits_child, dim=1)[:, 1].item()
            probability_parent = torch.softmax(logits_parent, dim=1)[:, 1].item()

            print(probability_parent, probability_child)

            linkability_threshold = 0.5
            if probability_parent < linkability_threshold:
                predicted_parent = 365
            if probability_child < linkability_threshold:
                predicted_child = 365

        return predicted_parent, predicted_child


    def save_model(self, foldername):
        # Create directory if it doesn't exist
        if not os.path.exists(foldername):
            os.makedirs(foldername)

        # Save the entire model, including child and parent classifiers
        model_path = f'{foldername}/model_checkpoint.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'parent_classifier_state_dict': self.parent_classifier.state_dict(),
            'optimizer_parent_state_dict': self.optimizer_parent.state_dict(),
            'child_classifier_state_dict': self.child_classifier.state_dict(),
            'optimizer_child_state_dict': self.optimizer_child.state_dict(),
            # 'loss': self.losses[-1],  # Save the last training loss
        }, model_path)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        # Load model and optimizer states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load child classifier and optimizer states
        self.child_classifier.load_state_dict(checkpoint['child_classifier_state_dict'])
        self.optimizer_child.load_state_dict(checkpoint['optimizer_child_state_dict'])

        # Load parent classifier and optimizer states
        self.parent_classifier.load_state_dict(checkpoint['parent_classifier_state_dict'])
        self.optimizer_parent.load_state_dict(checkpoint['optimizer_parent_state_dict'])


class ContextTokenizer:
    def __init__(self, tokenizer, max_length, special_tokens):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # special_tokens = {'additional_special_tokens': ['[Ms]','[Me]']}
        self.tokenizer.add_special_tokens(special_tokens)

    def tokenizeWcontext(self, row):

        mention = row["text"]
        left_context = row["left_context"]
        right_context = row["right_context"]

        if left_context or right_context:
            input_sequence = left_context + "[Ms]" + mention + "[Me]" + right_context
        else: 
            input_sequence = mention

        tokenizer_output = self.tokenizer(input_sequence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return tokenizer_output
    
class HierarchyDataset(Dataset):
    def __init__(self, tokenized_data, labels, special_token_masks):
        self.tokenized_data = tokenized_data
        self.labels = labels
        self.special_token_masks = special_token_masks

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, index):
        sample = self.tokenized_data[index]
        child_label, parent_label = self.labels[index]
        special_token_mask = self.special_token_masks[index]

        return {
            'input_ids': torch.Tensor(sample['input_ids']),
            'attention_mask': torch.Tensor(sample['attention_mask']),
            'token_type_ids': torch.Tensor(sample['token_type_ids']),
            'special_token_mask' : torch.Tensor(special_token_mask),
            'child_label': torch.tensor(child_label),
            'parent_label': torch.tensor(parent_label)
        }
        