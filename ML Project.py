# Step 1: Data Exploration and Preprocessing
# 1.1 Load the JSON data and convert it to a pandas DataFrame
import pandas as pd
import json

# Load the JSON data
file_path = 'path/to/your/dataset.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(df.head())

# 1.2 Remove duplicates and handle missing values
# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values (example: fill NaNs with empty strings)
df.fillna('', inplace=True)

# Display the DataFrame info to check for missing values
print(df.info())

# 1.3 Clean and normalize text data
import re

def clean_text(text):
    # Remove special characters and lowercase the text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

# Apply the clean_text function to the 'reviewText' and 'summary' columns
df['reviewText'] = df['reviewText'].apply(clean_text)
df['summary'] = df['summary'].apply(clean_text)

# Combine 'summary' and 'reviewText' fields for more context
df['combined_text'] = df['summary'] + ' ' + df['reviewText']

# Display the first few rows of the DataFrame
print(df[['combined_text']].head())


# 1.4 Create multi-label categories based on review content
# Example categories (you can define your own based on the dataset)
categories = ['Product Quality', 'Customer Service', 'Price', 'Functionality', 'Ease of Use']

# Function to create multi-label categories (this is a placeholder, you need to define your own logic)
def categorize_review(text):
    labels = []
    if 'quality' in text:
        labels.append('Product Quality')
    if 'service' in text:
        labels.append('Customer Service')
    if 'price' in text:
        labels.append('Price')
    if 'function' in text:
        labels.append('Functionality')
    if 'easy' in text:
        labels.append('Ease of Use')
    return labels

# Apply the categorize_review function to the 'combined_text' column
df['categories'] = df['combined_text'].apply(categorize_review)

# Display the first few rows of the DataFrame
print(df[['combined_text', 'categories']].head())

# 1.5 Tokenize and encode text using BERT tokenizer
from transformers import BertTokenizer

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and encode the text
df['input_ids'] = df['combined_text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Display the first few rows of the DataFrame
print(df[['combined_text', 'input_ids']].head())


# 1.6 Create multi-hot encoded label vectors for the categories
from sklearn.preprocessing import MultiLabelBinarizer

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=categories)

# Fit and transform the categories
df['label_vectors'] = list(mlb.fit_transform(df['categories']))

# Display the first few rows of the DataFrame
print(df[['categories', 'label_vectors']].head())

# Next Steps
# Handle class imbalance: Use techniques like oversampling or class weighting.
# Fine-tune a pre-trained BERT model: Use the transformers library to fine-tune BERT for multi-label classification.
# Implement data augmentation techniques: Improve model generalization.
# Experiment with different model architectures and hyperparameters: Optimize performance.
# Develop an ensemble model: Combine multiple models to improve overall performance.
# Create a simple API: Use Flask or FastAPI to showcase the model's capabilities.


# STEP 1:
# To handle class imbalance in your dataset, you can use techniques like oversampling the minority classes or applying class weights during model training. Here are two common approaches:
# 1. Oversampling the Minority Classes
# You can use the imblearn library's RandomOverSampler to oversample the minority classes.
# 2. Applying Class Weights During Model Training
# You can calculate class weights and pass them to the loss function during model training. This approach is useful when using libraries like PyTorch.

from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
import torch

# Initialize the RandomOverSampler
ros = RandomOverSampler()

# Resample the dataset
X_resampled, y_resampled = ros.fit_resample(df['input_ids'].tolist(), df['label_vectors'].tolist())

# Convert back to DataFrame
df_resampled = pd.DataFrame({'input_ids': X_resampled, 'label_vectors': y_resampled})

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(df_resampled['label_vectors']), y=df_resampled['label_vectors'])
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Define the loss function with class weights
loss_fn = torch.nn.BCEWithLogitsLoss(weight=class_weights)

# Display the first few rows of the resampled DataFrame
print(df_resampled.head())





# STEP 2:
To fine-tune a pre-trained BERT model for multi-label classification using the transformers library, follow these steps:

Step 1: Install Required Libraries
Make sure you have the required libraries installed:

pip install transformers torch scikit-learn

Step 2: Prepare the Dataset
Ensure your dataset is ready with tokenized inputs and multi-hot encoded labels, as shown in your previous steps.

Step 3: Create a Custom Dataset Class
Create a custom dataset class to handle the input data.

import torch
from torch.utils.data import Dataset

class TicketDataset(Dataset):
    def __init__(self, input_ids, label_vectors):
        self.input_ids = input_ids
        self.label_vectors = label_vectors

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_id = torch.tensor(self.input_ids[idx], dtype=torch.long)
        label_vector = torch.tensor(self.label_vectors[idx], dtype=torch.float)
        return input_id, label_vector


Step 4: Initialize the BERT Model for Multi-Label Classification
Use the BertForSequenceClassification model with num_labels set to the number of categories.

from transformers import BertForSequenceClassification, AdamW

# Number of categories
num_labels = len(categories)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


Step 5: Create Data Loaders
Create data loaders for training and validation.

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# Create the dataset
dataset = TicketDataset(df_resampled['input_ids'].tolist(), df_resampled['label_vectors'].tolist())

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)

Step 6: Define the Optimizer and Scheduler
Define the optimizer and learning rate scheduler.

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Define the learning rate scheduler
from transformers import get_linear_schedule_with_warmup

epochs = 4
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

Step 7: Train the Model
Train the model using the training data and evaluate it on the validation data.

from sklearn.metrics import f1_score, hamming_loss, accuracy_score, roc_auc_score

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs):
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch_input_ids, batch_labels = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_labels = batch_labels.to(device)

            model.zero_grad()
            outputs = model(batch_input_ids, labels=batch_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Training loss: {avg_train_loss}")

        # Validation
        model.eval()
        val_loss = 0
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                batch_input_ids, batch_labels = batch
                batch_input_ids = batch_input_ids.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_input_ids, labels=batch_labels)
                loss = outputs.loss
                val_loss += loss.item()

                logits = outputs.logits
                predictions.extend(torch.sigmoid(logits).cpu().numpy())
                true_labels.extend(batch_labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}, Validation loss: {avg_val_loss}")

        # Calculate evaluation metrics
        predictions = (np.array(predictions) > 0.5).astype(int)
        true_labels = np.array(true_labels)
        micro_f1 = f1_score(true_labels, predictions, average='micro')
        macro_f1 = f1_score(true_labels, predictions, average='macro')
        hamming = hamming_loss(true_labels, predictions)
        subset_acc = accuracy_score(true_labels, predictions)
        auc_roc = roc_auc_score(true_labels, predictions, average='macro', multi_class='ovr')

        print(f"Micro F1 Score: {micro_f1}")
        print(f"Macro F1 Score: {macro_f1}")
        print(f"Hamming Loss: {hamming}")
        print(f"Subset Accuracy: {subset_acc}")
        print(f"AUC-ROC: {auc_roc}")

# Train the model
train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs)


# Need to implement the following steps:
Implement data augmentation techniques: Improve model generalization.
Experiment with different model architectures and hyperparameters: Optimize performance.
Develop an ensemble model: Combine multiple models to improve overall performance.
Create a simple API: Use Flask or FastAPI to showcase the model's capabilities.