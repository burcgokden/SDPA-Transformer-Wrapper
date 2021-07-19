## Transformer with Scaled Dot Product Attention

This repository is the implementation of Scaled Dot  Product Attention (SDPA) Transformer model within a wrapper code suite for training and evaluation. The details of the transformer model can be found in [Transformer model for language understanding](https://www.tensorflow.org/text/tutorials/transformer)

This training and evaluation suite was used to train and evaluate a SDPA transformer model for Turkish-English and Portuguese-English translation tasks in the research article: [Power Law Graph Transformer for Machine Translation and Representation Learning](https://arxiv.org/abs/2107.02039)

#### Key Features

- Flexible model customization through a hyperparameter dictionary for SDPA Transformer model parameters.
- Simple interface for training the model with checkpoints at custom intervals, and highest accuracy observed.
- Early stopping after a number of epochs based on validation loss.
- Simple interface for evaluating trained model using BLEU score with greedy search and beam search.
- Data preparation framework for Neural Machine Translation for tensorflow datasets with capability to use a percentage of the train dataset or filter dataset based on a token  number in a sentence. 
- Capability to reverse source and target languages for input dataset.
- Keeps track of train and validation loss/accuracy for each epoch.

#### Sample Run:

Sample run trains and evaluates a 4-layer 8-head SDPA Transformer model for PT-EN translation task from tensorflow dataset found at: [ted_hrlr_translate/pt_to_en](https://www.tensorflow.org/datasets/catalog/ted_hrlr_translate#ted_hrlr_translatept_to_en)

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

#### Loss and Accuracy Curves:

Train and validation loss/accuracy values for each epoch are saved as pickle file and can be found in the train folder under save_model_path name:

```python
import common as cm

train_loss=cm.pklload("./model_saves/train/my_sdpa_transformer/train_loss.pkl")
val_loss=cm.pklload("./model_saves/train/my_sdpa_transformer/val_loss.pkl")
train_acc=cm.pklload("./model_saves/train/my_sdpa_transformer/train_accuracies.pkl")
val_acc=cm.pklload("./model_saves/train/my_sdpa_transformer/val_accuracies.pkl")
```

#### Single Instance Evaluation:

A sentence can be translated and compared to ground truth using greedy search only or beam search methods for single instance evaluation:

```python
e2e_model=sdpa_run.sdpa_transformer_e2e(
                                      tokenizer_obj_src = inp_obj.tokenizers_src,
                                      tokenizer_obj_tgt = inp_obj.tokenizers_tgt,
                                      checkpoint_path = './model_saves/',
                                      hpdict=hpdict_sdpa_transformer ,
                                      load_ckpt='train' # 'val' | 'valacc' | custom checkpoint path
                                     )

#greedy search only
translated_text, translated_tokens, _, eval_length = e2e_model.evaluate(sentence, max_length=50)
e2e_model.print_translation(sentence, translated_text, ground_truth, eval_length)

#beam search
translated_text_list, translated_tokens_list, tranlated_tokenid_list, eval_length = e2e_model.beam_evaluate(sentence, beam_size=4, max_length=50)
e2e_model.print_translation(sentence, translated_text_list[0], ground_truth, eval_length)
```

- Below sentences from test dataset are evaluated with beam length=4 by a model trained with same hyperparameters for a SDPA transformer detailed in the research article. Evaluation output may vary with each newly trained model.

> **Translating from:** perdemos o medo de criar uma coisa nova .  
> **Best probable translation:** we lost fear to create something new .  
> **Ground Truth:** we lost the fear of creating something new .  
>
> **Translating from:** vou mostrar aqui alguns exemplos , e vamos examinar alguns deles .  
> **Best probable translation:** let me show you here some examples , and let ' s examine some of them .  
> **Ground Truth:** i 'm going to show you some examples here , and we will run through some of them .  
>
> **Translating from:** ok , hoje quero falar sobre a forma como falamos do amor .  
> **Best probable translation:** okay , today i want to talk about how we talk about love .  
> **Ground Truth:** ok , so today i want to talk about how we talk about love .  
>
> **Translating from:** mas há uma grande diferença , isso só acontece dentro da colónia .  
> **Best probable translation:** but there ' s a big difference , it just happens inside the colony .  
> **Ground Truth:** but there 's a big difference , which is that it only happens within the colony .  
>
> **Translating from:** mas muito bons a absorver informação de muitas fontes diversas ao mesmo tempo .  
> **Best probable translation:** but very good at absorbing data from a lot of different sources at the same time .  
> **Ground Truth:** but they 're very good at taking in lots of information from lots of different sources at once .  
>
> **Translating from:** não podia construir isto com um anel de aço , da forma que sabia .  
> **Best probable translation:** i could n ' t build this up with a steel ring in the way i knew .  
> **Ground Truth:** i could n't build this with a steel ring , the way i knew .  
>
> **Translating from:** e gostaria de continuar a construir monumentos , que são amados por pessoas .  
> **Best probable translation:** and i ' d like to continue building monuments , which are loved by people .  
> **Ground Truth:** and i 'd like to keep building monuments that are beloved by people .  
>
> **Translating from:** a questão é que temos que ter um contexto , um limite para as nossas ações em tudo isto .  
> **Best probable translation:** the key thing is that we have to have a context , a range for our actions in all of this .  
> **Ground Truth:** the point is that we have to have a context , a gauge for our actions in all this .  
>
> **Translating from:** somos mais inteligentes , mais flexivéis , capazes de aprender mais , sobrevivemos em diferentes ambientes , emigrámos para povoar o mundo e viajámos até ao espaço .  
> **Best probable translation:** we ' re more intelligent , more flexible , we can learn more , we ' ve lived in different environments , we ' ve got into the world , and we push to space .  
> **Ground Truth:** we 're smarter , we 're more flexible , we can learn more , we survive in more different environments , we migrated to cover the world and even go to outer space .  
>
> **Translating from:** olhando para trás para os destroços e desespero daqueles anos , parece-me agora como se alguém tivesse morrido naquele lugar , e , no entanto , uma outra pessoa foi salva .  
> **Best probable translation:** looking behind the rubble and desperation of those years , it seems to me now as if someone died in that place , and yet another person was saved .  
> **Ground Truth:** now looking back on the wreckage and despair of those years , it seems to me now as if someone died in that place , and yet , someone else was saved .  
>
> **Translating from:** o cérebro pega em informação sem sentido e faz sentido a partir disso , o que significa que nunca vemos o que lá está , nunca vemos informação , só vemos o que nos foi útil ver no passado .  
> **Best probable translation:** so the brain takes information without it , and it makes sense out of it , which means that we never see what is there , we never see information , we only see what was useful for us to see in the past .  
> **Ground Truth:** right ? the brain takes meaningless information and makes meaning out of it , which means we never see what 's there , we never see information , we only ever see what was useful to see in the past .  
>
