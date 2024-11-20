import time
import argparse
import numpy as np
import torch
import evaluate

import json

from datasets import load_dataset
from transformers import T5Config, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer #DataCollatorWithPadding, 

import wandb
wandb.init(mode='disabled')

from modeling_t5dec import T5PseudoDecForConditionalGeneration

def run(tiny=False, batch_size=1, num_train_epochs=1, output_dir="./outputs/T5Dec", overwrite_output_dir=True) -> None:

    '''
    toy training script:  train the T5PseudoDec on wikitext
    see dataset: https://huggingface.co/datasets/wikitext
    script based on the following HF tutorial: https://huggingface.co/docs/transformers/tasks/sequence_classification
    #TODO: add system/hardware specs this ran on

    '''

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(1405) # pytorch random seed
    np.random.seed(1405) # numpy random seed
    torch.backends.cudnn.deterministic = True

    #set test metrics
    accuracy = evaluate.load("accuracy")
    def _compute_metrics(eval_pred): 
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    #choose dataset option
    tiny=tiny
    if tiny:
        output_dir= output_dir + "/wikitext-2-raw-v1" 
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    else: 
        print("Full Wikitext!")
        output_dir= output_dir +"/wikitext-103-raw-v1"  
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1") 
    
    #create dataset
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    def _tokenize(examples):
        tokens = tokenizer(examples["text"], truncation=True, padding=True, return_tensors= 'pt')
        #tokens['labels'] = copy.deepcopy(tokens['input_ids'])
        return tokens #TODO: DEBUG for batch size >1. ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`labels` in this case) have excessive nesting (inputs type `list` where type `int` is expected).
    
    tokenized_dataset = dataset.map(_tokenize, batched=True)
    #tokenized_dataset = tokenized_dataset.map(lambda examples: {"labels": examples["input_ids"]}, batched=True) #TODO: debug
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) #see https://huggingface.co/docs/transformers/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling

    #set model specs
    try: 
        with open("your_model_config.json", "r") as f: #to write custom see https://huggingface.co/google-t5/t5-small/blob/main/config.json
            config = json.load(f)
    except (FileNotFoundError):
        print("No model config file found! Using T5-small defaults.")
        config=T5Config() #yields t5-small architecture per default
        config.decoder_start_token_id = 0

    model = T5PseudoDecForConditionalGeneration(config=config)

    #set up Trainer
    training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4, #NOTE: gradient accumulation increases effective batch size but introduces slowdown (e.g. forces 'use_cache = False'), disable if not needed
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit", #see: https://huggingface.co/docs/transformers/v4.40.2/en/perf_train_gpu_one
    num_train_epochs=num_train_epochs,
    evaluation_strategy="epoch",
    eval_do_concat_batches=False, #else will always OOM, see HF issue: https://discuss.huggingface.co/t/prohibitively-large-ram-consumption-on-trainer-validation/83486
    save_strategy="epoch",
    load_best_model_at_end=True,
    #push_to_hub=True,
)
    
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=_compute_metrics,
)
    #train the model
    trainer.train()

    #save the model
    trainer.save_model()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Trains T5Dec on Wikitext from scratch.') 
    parser.add_argument('-t', '--tiny', type=bool, help='whether to use the small version of wikitext',
                        required=False, default=False)
    parser.add_argument('-o', '--output', type=str, help='output directory for trained model',
                        required=False, default="./outputs/T5Dec")
    parser.add_argument('-b', '--batch', type=int, help='set the batch size for train, test',
                    required=False, default=1)
    parser.add_argument('-e', '--epoch', type=int, help='set the number of training epochs',
                required=False, default=1)
    parser.add_argument('-r', '--rewrite', type=bool, help='overwrite output directory',
            required=False, default=True)
    
    args = parser.parse_args()
    st = time.time()

    run(tiny=args.tiny, output_dir=args.output, batch_size=args.batch, num_train_epochs=args.epoch, overwrite_output_dir=args.rewrite)

    elapsed_time = time.time() - st
    print('Avg. execution time per epoch:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time/args.epoch)))

#TODO: add requirements.txt to this repo
#TODO: cleanup all depreciated code
#TODO: write shell script for slum, add *.sh to .gitignore


