import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset


df = pd.read_feather('/content/drive/MyDrive/testdata/pressreleases.feather')

# sample for testing
#df = df.sample(n = 100)


train_df = df.sample(frac = 0.75)

# Creating dataframe with
# rest of the 25% values
eval_df = df.drop(train_df.index)


train_data = train_df['text_pre-proc'].tolist()
train_label = train_df['label'].tolist()
valid_data = eval_df['text_pre-proc'].tolist()
valid_label = eval_df['label'].tolist()


# Step 1: Load and tokenize the data
# Replace the following with your own data loading and preprocessing code
train_texts = train_data # List of training text sentences
train_labels = train_label # List of corresponding labels (e.g., 0 or 1 for binary classification)

# Map the string labels to numerical indices
label_to_index = {label: i for i, label in enumerate(set(train_labels))}
train_labels = [label_to_index[label] for label in train_labels]

# Step 2: Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the input texts and convert them to tensors
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')

# Convert labels to tensors
train_labels = torch.tensor(train_labels)

# Step 3: Load the pre-trained word embedding model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_index))

# Step 4: Create the classifier model
# The classifier is already integrated into the pre-trained BERT model

# Step 5: Train the classifier
# Define the training parameters (learning rate, batch size, etc.)
learning_rate = 1e-5
batch_size = 32
num_epochs = 5

# Create DataLoader for the training data
train_dataset = TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Step 6: Evaluate the classifier (Optional)
# You can evaluate the model's performance on a validation or test dataset.
# Replace the following with your own validation/test data and evaluation code.
validation_texts = valid_data  # List of validation text sentences
validation_labels = valid_label # List of corresponding labels
validation_encodings = tokenizer(validation_texts, truncation=True, padding=True, return_tensors='pt')
validation_labels = torch.tensor([label_to_index[label] for label in validation_labels])

validation_dataset = TensorDataset(validation_encodings.input_ids, validation_encodings.attention_mask, validation_labels)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in validation_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")

    