## Transformer with Scaled Dot Product Attention

This repository is the implementation of Scaled Dot  Product Attention (SDPA) Transformer model within a wrapper code suite for training and evaluation. The details of the transformer model can be found in [Transformer model for language understanding](https://www.tensorflow.org/text/tutorials/transformer)

This training and evaluation suite was used to evaluate a SDPA transformer model for Turkish-English and Portuguese-English translation tasks in the research article: [Power Law Graph Transformer for Machine Translation and Representation Learning](https://github.com/burcgokden/Power-Law-Graph-Transformer/blob/main/plgt_paper.pdf)

#### Key Features

- Flexible model customization through a hyperparameter dictionary for SDPA Transformer model parameters.
- Simple interface for training the model with checkpoints at custom intervals, and highest accuracy observed.
- Early stopping after a number of epochs based on validation loss.
- Simple interface for evaluating trained model using BLEU score with greedy search and beam search.
- Data preparation framework for Neural Machine Translation for tensorflow datasets with capability to use a percentage of the train dataset or filter dataset based on a token  number in a sentence. 
- Capability to reverse source and target languages for input dataset.

#### Sample Run:

Sample run trains and evaluates a 4-layer 8-head SDPA Transformer model using a PT-EN translation task from tensorflow dataset found at: [ted_hrlr_translate/pt_to_en](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en)

The tokenizer model is built using BERT Subword Tokenizer for Machine Translation implemented at [BERT Subword Tokenizer for Machine Translation](https://github.com/burcgokden/BERT-Subword-Tokenizer-Wrapper)

- Prepare the Dataset for Machine Translation:

```python
import nmt_data_prep as ndp

inp_obj = ndp.src_tgt_data_prep(
                 src_lang='pt',
                 tgt_lang='en',
                 BUFFER_SIZE=20000,
                 BATCH_SIZE = 64,
                 dataset_file="ted_hrlr_translate/pt_to_en",
                 load_dataset=True,
                 train_percent=None,
                 model_name = "./ted_hrlr_translate_pt_en_tokenizer",
                 revert_order=False,
                 shuffle_set=True,
                 shuffle_files=True,
                 MAX_LENGTH=None, 
                 verbose=True)
```

- Define hyperparameter dictionary for SDPA Transformer:

```python
hpdict_sdpa_transformer = {
          "num_layers":4,
          "d_model": 512,
          "num_heads": 8,
          "dropout_rate": 0.1,
          "dff": 2048,
          "input_vocab_size": inp_obj.tokenizers_src.get_vocab_size(),
          "target_vocab_size": inp_obj.tokenizers_tgt.get_vocab_size(),
          "pe_input": 1000,
          "pe_target": 1000,
          "epochs": 40,
          "save_model_path": "my_sdpa_transformer",       
          "early_stop_threshold": 4.0,
          "early_stop_counter": 10,
          "early_stop_accuracy": 0.59,
          "warmup_steps": 4000
          }
```

- Initialize the end-to-end model training suite and run train:
```python
import sdpa_transformer_run_model as sdpa_run

e2e_obj=sdpa_run.sdpa_transformer_e2e(
                                      tokenizer_obj_src = inp_obj.tokenizers_src,
                                      tokenizer_obj_tgt = inp_obj.tokenizers_tgt,
                                      checkpoint_path = './model_saves/',
                                      hpdict=hpdict_sdpa_transformer ,
                                      load_ckpt=None,
                                     )

train_loss, train_accuracy, val_loss, val_accuracy=e2e_obj.train_model(
                                                                       inp_obj.train_batches, 
                                                                       inp_obj.val_batches,
                                                                       chkpt_epochs=[24, 30]
                                                                      )

```

- Evaluate the trained SDPA Model using greedy or beam search:

```python
import sdpa_evaluate_bleu_score as sebg

#greedy search only
sebg.evaluate_bleu(
                 model_dict=hpdict_sdpa_transformer,
                 model_name="my_sdpa_transformer",
                 model_type='train',
                 src_lang='pt',
                 tgt_lang='en',
                 dataset_file="ted_hrlr_translate/pt_to_en",
                 revert_order=False,
                 inp_obj=None,
                 chkpt_path= './model_saves/',
                 data_path= './model_data/',              
                 load_ckpt='train', # 'val' | 'valacc' | custom checkpoint path
                 tok_model_name="./ted_hrlr_translate_pt_en_tokenizer",
                 max_length=50,  #offset to evaluate model beyond input sentence length
                 ds_max_length=None, #None for no filtering input sentence length
                 verbose=True
                )

#beam search
sebg.beam_evaluate_bleu(
                 model_dict=hpdict_sdpa_transformer,
                 beam_size=4,
                 model_name="my_sdpa_transformer",
                 model_type='train',
                 src_lang='pt',
                 tgt_lang='en',
                 dataset_file="ted_hrlr_translate/pt_to_en",
                 revert_order=False,
                 inp_obj=None,
                 chkpt_path= './model_saves/',
                 data_path= './model_data/',              
                 load_ckpt='train',
                 tok_model_name="./ted_hrlr_translate_pt_en_tokenizer",
                 max_length=50,
                 ds_max_length=None,
                 verbose=True
                )

```
                              
                           


