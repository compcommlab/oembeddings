### import needed datasets 
import pandas as pd
from datasets import load_dataset #, load_metric, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer #, EvalPrediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

### dataset loop
datasets = ['facebook', 'twitter', 'nationalrat', 'pressreleases']

for d in datasets:
    ### reformat the data so that it fits the huggingface architecture
    # load dataset
    df = pd.read_feather(f'/Users/janabernhard/Documents/Projekte/2023_Embedding/code/data_preproc/{d}.feather', columns=['label','text_pre-proc'])
    # rename columns & make the labels strings
    df.rename(columns={'text_pre-proc': 'text', 'label':'labels'}, inplace=True)
    mapping = {'SPOE': 0, 'OEVP': 1, 'FPOE': 2, 'NEOS': 3, 'GRUE':4}
    df['labels'] = df['labels'].replace(mapping)
    # drop NAs (we don't have any but just in case)
    df = df.dropna()
    # draw sample for pre-testing 
    #df = df.sample(n=100)
    # save to csv
    df.to_csv(f'{d}.csv', index=False)


    ### Load the CSV dataset
    csv_path = f'{d}.csv'
    dataset = load_dataset('csv', data_files=csv_path)

    # tokenize the pre-processed data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Use the 'train' split
    train_dataset = tokenized_dataset["train"].shuffle(seed=42)

    # Split the 'train' data into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    validation_size = len(train_dataset) - train_size

    small_train_dataset, small_validation_dataset = train_dataset.train_test_split(
        test_size=validation_size, shuffle=True, seed=42)

    # loop through models 
    models = ['xlm-roberta-large', 'distilbert-base-multilingual-cased', 'uklfr/gottbert-base', 'microsoft/mdeberta-v3-base']
    results = {}

    for m in models:
        model = AutoModelForSequenceClassification.from_pretrained(m, num_labels=5)
        training_args = TrainingArguments(output_dir="test_trainer")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
        )
        
        # Load the validation dataset from the CSV file
        validation_csv_path = 'pressreleases.csv'  # Update with the correct path
        validation_dataset = load_dataset('csv', data_files=validation_csv_path)['train']
        
        # Tokenize the validation dataset
        tokenized_validation_dataset = validation_dataset.map(
            lambda example: tokenizer(example["text"], padding="max_length", truncation=True),
            batched=True,
        )
        
        # Evaluate the model on the validation dataset
        eval_results = trainer.predict(tokenized_validation_dataset)
        
        # Calculate metrics
        predictions = eval_results.predictions.argmax(axis=1)
        ground_truth = tokenized_validation_dataset['labels']
        
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions, average='weighted')
        recall = recall_score(ground_truth, predictions, average='weighted')
        f1_macro = f1_score(ground_truth, predictions, average='macro')

        # save metrics for json output
        results[m] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Macro": f1_macro
        }
        
        # Print the evaluation results
        print(f"Model: {m}")
        print('\n')
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Macro: {f1_macro}")

    # create one json per dataset
    with open(f'evaluation_results_{d}.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)