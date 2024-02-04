import torch
from tqdm import tqdm
import os
import datetime
from torch.utils.data import Dataset

class EntityMentionTokenizer:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

        special_tokens = {'additional_special_tokens': ['[Ms]','[Me]', '[ENT]']}
        self.tokenizer.add_special_tokens(special_tokens)

    def tokenizeWcontext(self, row):

        mention = row["mention"]
        left_context = row["context_left"]
        right_context = row["context_right"]

        if left_context or right_context:
            input_sequence = left_context + "[Ms]" + mention + "[Me]" + right_context
        else: 
            input_sequence = mention

        tokenizer_output = self.tokenizer(input_sequence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

        return tokenizer_output
    
    def tokenize_entity_description(self, row):

        title = row['code']
        description=row['description']

        input_sequence= title + "[ENT]" + description

        tokenizer_output = self.tokenizer(input_sequence, padding = 'max_length', truncation = True, max_length=self.max_length)

        return tokenizer_output

class EntityMentionPooling(torch.nn.Module):
    def __init__(self, special_token_type, pool_type="average"):
        super(EntityMentionPooling, self).__init__()
        self.pooling_type = pool_type
        self.special_token_type = special_token_type

    def forward(self, sequence_embeddings, special_tokens_mask):
        # special token type can be 'mention' or 'entity'

        pooled_representations = []

        mask_indices = special_tokens_mask.nonzero(as_tuple=True)[-1]

        if self.special_token_type == 'mention':
        
            # Reshape the mask indices into pairs of [Ms] and [Me]
            mask_indices_pairs = mask_indices.view(-1, 2)

            for (ms_index, me_index), batch_embedding in zip(mask_indices_pairs, sequence_embeddings):
                # Extract specified embeddings
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


        if self.special_token_type == 'entity':
            
            for ent_index, batch_embedding in zip(mask_indices, sequence_embeddings):
                indexed_embeddings = batch_embedding[ent_index, :]

                pooled_representations.append(indexed_embeddings)
            final_representation = torch.stack(pooled_representations, dim=0)

        return final_representation
    
class ICD10BiEncoder:
    def __init__(self, mention_transformer, entity_trasformer, optimizer_mention, optimizer_entity):

        # define the two transformers
        self.mention_transformer = mention_transformer
        self.entity_transformer = entity_trasformer

        self.mention_optimizer = optimizer_mention
        self.entity_optimizer = optimizer_entity


        self.train_losses = []
        self.val_losses = []

    def validation(self, val_dataloader, mention_pooling, entity_pooling, checkpoint_folder, epoch):

        self.mention_transformer.eval()
        self.entity_transformer.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f'Validation', leave=True):

                # Extract tensors from the batch dictionary
                mention_input_ids = batch['mention_input_ids'].squeeze(1)
                mention_attention_mask = batch['mention_attention_mask'].squeeze(1)
                mention_token_type_ids = batch['mention_token_type_ids'].squeeze(1)

                entity_input_ids = batch['entity_input_ids'].squeeze(1)
                entity_attention_mask = batch['entity_attention_mask'].squeeze(1)
                entity_token_type_ids = batch['entity_token_type_ids'].squeeze(1)

                mention_special_tokens_mask = batch['mention_special_token_mask']
                entity_special_tokens_mask = batch['entity_special_token_mask']

                # forward pass
                mention_outputs = self.mention_transformer(input_ids=mention_input_ids, attention_mask=mention_attention_mask, token_type_ids=mention_token_type_ids)
                entity_outputs = self.entity_transformer(entity_input_ids, token_type_ids=entity_token_type_ids, attention_mask=entity_attention_mask)

                if mention_pooling == 'CLS':
                    mention_embeddings = mention_outputs.last_hidden_state[:, 0, :]
                    entity_embeddings = entity_outputs.last_hidden_state[:, 0, :]
                else:
                    # apply pooling fucntion
                    mention_embeddings = mention_pooling(mention_outputs.last_hidden_state, mention_special_tokens_mask)
                    entity_embeddings = entity_pooling(entity_outputs.last_hidden_state, entity_special_tokens_mask)
                    
                # compute loss
                    
                # initialize a tensor to store similarity scores
                num_mentions = mention_embeddings.size(0)
                num_entities = entity_embeddings.size(0)

                # Compute similarity scores using matrix multiplication
                similarity_scores = torch.matmul(mention_embeddings, entity_embeddings.t())

                # Compute the loss based on the described formula
                correct_entity_scores = similarity_scores[range(num_mentions), range(num_entities)]

                # Compute the overall loss
                returned_loss = -correct_entity_scores + torch.logsumexp(similarity_scores, dim=1)

                loss = returned_loss.mean()
                total_val_loss += loss

            average_loss = total_val_loss / len(val_dataloader)

                # store training metrics
            self.val_losses.append(average_loss)

            # Print or log training statistics for the epoch
            print(f'Validation Loss: {loss.item()}')

            checkpoint_folder = checkpoint_folder
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)

            # unique checkpoint path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(checkpoint_folder, f'checkpoint_epoch_{epoch}_{timestamp}.pt')

            torch.save({
                'epoch': epoch,
                'entity_model_state_dict': self.entity_transformer.state_dict(),
                'mention_model_state_dict': self.mention_transformer.state_dict(),
                'entity_optimizer_state_dict': self.entity_optimizer.state_dict(),
                'mention_optimizer_state_dict': self.mention_optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses' : self.val_losses
                # Add other relevant information
            }, checkpoint_path)

    def train(self, train_dataloader, val_dataloader, num_epochs, checkpoint_folder, mention_pooling, entity_pooling):

        for epoch in range(num_epochs):
            total_loss = 0.0
            losses_list = []

            self.mention_transformer.train()
            self.entity_transformer.train()

            for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=True):

                self.mention_optimizer.zero_grad()
                self.entity_optimizer.zero_grad()
                 
                # Extract tensors from the batch dictionary
                mention_input_ids = batch['mention_input_ids'].squeeze(1)
                mention_attention_mask = batch['mention_attention_mask'].squeeze(1)
                mention_token_type_ids = batch['mention_token_type_ids'].squeeze(1)

                entity_input_ids = batch['entity_input_ids'].squeeze(1)
                entity_attention_mask = batch['entity_attention_mask'].squeeze(1)
                entity_token_type_ids = batch['entity_token_type_ids'].squeeze(1)

                mention_special_tokens_mask = batch['mention_special_token_mask']
                entity_special_tokens_mask = batch['entity_special_token_mask']

                # forward pass
                mention_outputs = self.mention_transformer(input_ids=mention_input_ids, attention_mask=mention_attention_mask, token_type_ids=mention_token_type_ids)
                entity_outputs = self.entity_transformer(entity_input_ids, token_type_ids=entity_token_type_ids, attention_mask=entity_attention_mask)

                if mention_pooling == 'CLS':
                    mention_embeddings = mention_outputs.last_hidden_state[:, 0, :]
                    entity_embeddings = entity_outputs.last_hidden_state[:, 0, :]
                else:
                    # apply pooling fucntion
                    mention_embeddings = mention_pooling(mention_outputs.last_hidden_state, mention_special_tokens_mask)
                    entity_embeddings = entity_pooling(entity_outputs.last_hidden_state, entity_special_tokens_mask)
                    
                # compute loss
                    
                # initialize a tensor to store similarity scores
                num_mentions = mention_embeddings.size(0)
                num_entities = entity_embeddings.size(0)

                # Compute similarity scores using matrix multiplication
                similarity_scores = torch.matmul(mention_embeddings, entity_embeddings.t())

                # Compute the loss based on the described formula
                correct_entity_scores = similarity_scores[range(num_mentions), range(num_entities)]

                # Compute the overall loss
                returned_loss = -correct_entity_scores + torch.logsumexp(similarity_scores, dim=1)

                losses_list.append(returned_loss)
                loss = returned_loss.mean()
                total_loss += loss

                loss.backward()
                self.mention_optimizer.step()
                self.entity_optimizer.step()


            average_loss = total_loss / len(train_dataloader)

            # store training metrics
            self.train_losses.append(average_loss)

            # Print or log training statistics for the epoch
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss:.4f}")

            # call val
            self.validation(val_dataloader, mention_pooling, entity_pooling, checkpoint_folder, epoch)

    def get_entity_embeddings(self, entity_loader, entity_pooling):

        embeddings = []

        for entities in entity_loader:

            entity_input_ids = entities['entity_input_ids']
            entity_attention_mask = entities['entity_attention_mask']
            entity_token_type_ids = entities['entity_token_type_ids']
            entity_special_tokens_mask = entities['entity_special_token_mask']

            entity_outputs = self.entity_transformer(input_ids=entity_input_ids, 
                                                        attention_mask=entity_attention_mask, 
                                                        token_type_ids=entity_token_type_ids)
            
            entity_embeddings = entity_pooling(entity_outputs.last_hidden_state, entity_special_tokens_mask)

            embeddings.extend(entity_embeddings)

        return embeddings

class EntityMentionDataset(Dataset):
    def __init__(self, mention_data, entity_data, mention_special_token_masks, entity_special_token_masks):
        self.mention_data = mention_data
        self.entity_data = entity_data
        self.mention_special_token_masks = mention_special_token_masks
        self.entity_special_token_masks = entity_special_token_masks

    def __len__(self):
        return len(self.mention_data)

    def __getitem__(self, index):
        mentions = self.mention_data[index]
        entities = self.entity_data[index]
        mention_special_token_masks = self.mention_special_token_masks[index]
        entity_special_token_masks = self.entity_special_token_masks[index]

        return {
            'mention_input_ids': torch.as_tensor(mentions['input_ids']).clone().detach().long(),
            'entity_input_ids': torch.as_tensor(entities['input_ids']).clone().detach().long(),
            'mention_attention_mask': torch.as_tensor(mentions['attention_mask']).clone().detach().long(),
            'entity_attention_mask': torch.as_tensor(entities['attention_mask']).clone().detach().long(),
            'mention_token_type_ids': torch.as_tensor(mentions['token_type_ids']).clone().detach().long(),
            'entity_token_type_ids': torch.as_tensor(entities['token_type_ids']).clone().detach().long(),
            'mention_special_token_mask': torch.as_tensor(mention_special_token_masks).clone().detach().long(),
            'entity_special_token_mask': torch.as_tensor(entity_special_token_masks).clone().detach().long(),
        }
