# T5PseudoDecoderOnly
A 'pseudo' decoder-only model with a language head, based on the [official huggingface T5ForConditionalGeneration](https://github.com/huggingface/transformers/blob/cbe58b4269457a6ca66a556224b23f9ef246f905/src/transformers/models/t5/modeling_t5.py#L1554). Supports traditional causal language modelling as well as ['non-causal'](https://arxiv.org/abs/2204.05832) seq2seq training. 

### Why 'Pseudo' Decoder-Only?
This model inherits from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5), originally presented in [Raffel et. al (2019)](https://arxiv.org/abs/1910.10683), which is an encoder-decoder model. For simplicity, I have only replaced the encoder `T5Stack` with a `T5PseudoEncoder`, which is simply an embedding layer followed by a layer norm. The weights of the embedding layer are tied to the standard shared token embedding layer of the model. 

If set to language modeling, the model does causal cross attention over the shifted input tokens, i.e. it sees one token at a time. If set to 'non-causal' seq2seq the model performs non-causal cross-attention over the whole input, as the original T5 model. Given that the weights in the embedding layer in the pseudo-encoder are tied to the one in the decoder, the model has no way to learn an additional input representation in the encoder, and therefore relies on the decoder only.

When generating, this model applies non-causal attention over all given input-tokens.

**Given that this follows the framework of T5, if you query `model.config.is_encoder_decoder` you will always get `True` !**

---

### How to Use It

#### Set up the Environment

    >>> conda create -n t5dec python=3.10 pip
    >>> # .... environment is being created
    >>> conda activate t5dec
    >>> pip install -r t5dec_requirements.txt

#### Load the Model    

This model can be initialised with pretrained weights from the huggingface T5 transformer, and should seamlessly be usable with huggingface and pytorch/lightning.


    >>> from transformers import AutoTokenizer
    >>> from modeling_t5dec import T5PseudoDecForConditionalGeneration, T5PseudoEncoder, T5_DecPreTrainedModel 

    >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    >>> model = T5PseudoDecForConditionalGeneration.from_pretrained("google-t5/t5-small")

Alternatively, you can train the model from scratch yourself, e.g. on [Wikitext](https://huggingface.co/datasets/wikitext). See the example script `train.py` described below. 

---    

#### Train the Model

##### Example Training Script for Autoregressive LM
This repo comes with a toy training script to train the model to generate Wikipedia text spans in a purely autoregressive manner. Please be aware that all hyperparameters in said script are not optimised in any way. Without the 'tiny' flag, the model defaults to train on wikitext-103. 
You should be able to adjust the script to every dataset that only contains text (no additional labels) with minimal effort.  
Credit: The structure of the script is based on [this huggingface tutorial on sequence classification with BERT](https://huggingface.co/docs/transformers/tasks/sequence_classification). 


    >>> python train.py --tiny=True --output="./your_custom_output_dir" --batch=16 --epoch=2 

In case you are training your T5Dec model on a slurm cluster, this repo also contains `slurm_train_conda.sh` for your convenience.  
If you then plan to share the trained model on the huggingface hub, there is `push_to_hub.py`, which contains the most important commands for uploading your model.    

##### Manual Training Pipelines
To set the model to traditional autoregressive language modeling, either hand it `labels` that are exactly equal to the `input_ids`, or manually set the `causal_LM` flag to `True` in the training call. 

    >>> #Auto-detect
    >>> input_ids = tokenizer(
        "Studies have shown that owning a dog is good for you.", return_tensors="pt"
        ).input_ids
    >>> labels = copy.deepcopy(input_ids)
    >>> outputs = self.model(input_ids, labels)

    >>> #Manually setting to causal LM
    >>> # this is necessary if your tokenizer pads your inputs for batch 
    >>> processing and you then set labels to -100 at the location of pad_token/generally choose to set some values to -100 in the labels.
    >>> # ... set your labels how you want somewhere else ...
    >>>
    >>> outputs = self.model(**batch, causal_LM = True)

This model is set to use traditional pytorch CE loss. The reduction method can directly be set in the training call. The methods available directly follow the pytorch CE loss implementation, i.e. 'none', 'sum', and 'mean'. 
If you manually set a loss type, the model will automatically set the loss on padding tokens to zero for you.  
If the flag is not used, the model defaults to mean-reduction, and no weights.

    >>> outputs = self.model(**batch, reduce_loss='sum')

This model comes with nominal support for attention head pruning in its decoder stack, adapted from the [functionality of the same name](https://github.com/huggingface/transformers/blob/cbe58b4269457a6ca66a556224b23f9ef246f905/src/transformers/models/t5/modeling_t5.py#L1942) in the huggingface `T5Encoder`. However, I have not tested this function *at all*. So if you find yourself using it, feel free to share results of potential debugging/whether it worked for you. 

##### Note: On T5 Overfitting
T5 is not that large of a model, and T5Dec is only half it's size, so maybe don't overdo it with the epochs/make sure to cross-validate.

## Known Issues: 
### Not Usable Without Finetuning. 
Expect gibberish if you use this out-of-the-box with weights from the hub. This happens because these pretrained T5 decoder weights expect a proper task-dependent input representation in latent space from the encoder, which we obviously do not want to provide. I sadly do not have access to the hardware necessary to train this model to similar performance as the other standard T5 models on the hub. **Feel free to solve this problem for the community if you have the motivation and the hardware.**

### T5 Numerical Instability Issues
By virtue of using the T5Stack class as its decoder, this model inherits its problems (as of May 9th 2024). This means that when finetuning using weights `from_pretrained` for T5-base, and -large (which were pretrained using bf16), use bf16-able hardware, e.g NVIDIA AX40, RTX A6000, or else risk numerical overflow. If no bf16-able hardware available, and you see logits/loss/probs becoming nan or inf, finetune in fp32.
See this [**shelved** pull request on huggingface's T5](https://github.com/huggingface/transformers/pull/10956) and [this comment on how to train](https://github.com/huggingface/accelerate/issues/243#issuecomment-1236975399).

### Loading Model Checkpoints
This model was trained with a weigh-decay optimizer, so when you load a model checkpoint, make sure that you load the general checkpoint (which also means the optimizer dict) and not just the model state dict. See [this pytorch tutorial](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html) and [this stackoverflow post](https://stackoverflow.com/questions/71693736/pytorch-torch-load-load-checkpoint-and-learning-rate). 

## Related Projects

[decoder-only T5, implemented in JAX](https://github.com/google/flaxformer/blob/ea17eb012a1d340ddff017b7a534c2162aaec34c/flaxformer/architectures/t5/t5_architecture.py#L1484); part of Google's flaxformer library. 


