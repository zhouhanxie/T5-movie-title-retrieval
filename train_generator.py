import pandas as pd
from datasets import Dataset
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_DATASETS_CACHE"]="./huggingface_cache"
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_utils import CustomTrainer, Trainer, CustomTrainingArguments
from transformers import TrainingArguments
from transformers import  DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback, TrainerCallback
from huggingface_utils import ProgressCallback
from sklearn.model_selection import train_test_split


def to_bitfit(model, verbose=False):
    """
    turn off anything except bias and classification head 
    in a transformer model
    """
    if verbose:
        print('most parameters will be turned off.')
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.lm_head.named_parameters():
        param.requires_grad = True
        
    return model

def main(args):
    data = pd.read_csv(args.data_path)
    data['input'] = ['Perform entity recognition and extract movie titles from the following text: \n'+i for i in data['input']]

    train, test = train_test_split(data, test_size=0.1, random_state=0)
    train_df = pd.DataFrame({'source':list(train['input'].astype(str)), 'target':list(train['output'].astype(str))})
    train_dataset = Dataset.from_pandas(train_df)
    test_df = pd.DataFrame({'source':list(test['input']), 'target':list(test['output'])})
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

    def preprocess_function(examples):
        inputs = examples['source']
        targets = examples['target']
        model_inputs = tokenizer(text = inputs, max_length=148, padding='max_length', truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(text_target=targets, max_length=50, padding='max_length', truncation=True)
            
        labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in labels['input_ids']] 
                ]

        model_inputs["labels"] = labels["input_ids"][0] # [0] for 2d->1d
        
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function)
    test_dataset = test_dataset.map(preprocess_function)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name_or_path)

    # if you really can't fit your model in try bitfit, which only adjusted bias terms
    # model = to_bitfit(model, verbose=True)

    data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
        )

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset, 
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=CustomTrainingArguments(
            load_best_model_at_end = True,
            output_dir = args.output_dir,
            save_strategy = 'epoch',
            evaluation_strategy = 'epoch',
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            save_total_limit =1,
            gradient_accumulation_steps = args.gradient_accumulation_steps,
            logging_steps=1
        )
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience = args.tolerance_steps,
        early_stopping_threshold=1e-7
    )
    trainer.add_callback(early_stopping_callback)
    trainer.train()
    trainer.save_model()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generator Model')
    
    from easydict import EasyDict as edict
    args = dict(
        data_path='./data/gpt3_annotated_generative_movie_title_ner/gpt_tagged_submissions.csv', 
        base_model_name_or_path="google/flan-t5-xl",
        output_dir='./checkpoints/generative_movie_title_ner',
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=12,
        tolerance_steps=1,
        learning_rate=3e-5
    )
    args = edict(args)
        
    
    main(args)
