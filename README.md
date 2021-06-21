# Sibyl

Sibyl is a tool for generating new data from what you have already. Transform your data in over 35 different ways by either selecting one or sampling `n` transformations at a time. 

There are two primary kinds of transformations:
- Invariant (INV) : transform the input, but the expected output remains the same.
  - ex. "I love NY" + `Emojify` = "I 💗 NY", which has an INV effect on the sentiment.
- Sibylvariant (SIB) : transform the input and the expected output may change in some way.
  - ex. "I love NY" + `ChangeAntonym` = "I hate NY", which has a SIB effect on the sentiment (label inverted from positive (1) to negative (0)).
  
Some transformations also result in *soft labels* and they're called SIB-mix transformations. For example, within topic classification you could use `SentMix` on your data to randomly combine two inputs from different classes into one input and then shuffle the sentences around. This new data would have a new label with probabilities weighted according to the relative length of text contributed by the two inputs. Illustrating with `AG_NEWS`, if you mix an article about sports with one about business, it might result in a new soft label like [0, 0.5, 0.5, 0]. The intuition here is that humans would be able to recognize that there are two topics in a document and we should expect our models to behave similarly (i.e. the model predictions should be very close to 50/50 on the expected topics). 

## Pretrained Models

We offer links to several of pre-trained + fine-tuned BERT models used during our evaluation. 

- [bert-base-uncased-sst2-INV](https://drive.google.com/file/d/17rytP7-qSJbgD1r3cpvgqr1JO-viV09V/view?usp=sharing)
- [bert-base-uncased-sst2-SIB](https://drive.google.com/file/d/18XPJxlkuOn5gHQaBZK9NVRxqv1CRd-xE/view?usp=sharing)
- [bert-base-uncased-ag_news-INV](https://drive.google.com/file/d/1Z7lqIrFh8pE7_Rh4R_eBPvtutYWTAecL/view?usp=sharing)
- [bert-base-uncased-ag_news-SIB-mix](https://drive.google.com/file/d/1zIIoQbqZqqvSFhp6ovSv6szrDzYUHlMr/view?usp=sharing)

## Examples

Here's a quick example of using a single transform:

```
from transforms import HomoglyphSwap

transform = HomoglyphSwap(change=0.75)
string_in = "The quick brown fox jumps over the lazy dog"
string_out = transform(string_in)
print(string_out) 

>> Tհe quіc𝒌 Ьⲅоԝn 𝚏о× ϳuｍрѕ оѵеⲅ 𝚝հе ⅼɑzу ԁoɡ
```

Here's a quick example using `transform_dataset` which uniformly samples from the taxonomized transformations so that you generate new data relevant to your particular task. 

```
from datasets import load_dataset
from transforms import *
from utils import *

dataset = load_dataset('glue', 'sst2')
train_data = dataset['train']
train_data.rename_column_('sentence', 'text')

task = 'sentiment'
tran = 'SIB'
n = 2

out = transform_dataset(
    train_data[:5], 
    num_transforms=n, 
    task=task, 
    tran=tran
)

new_text, new_label, trans_applied = out
```

Here are some examples we've already prepared:

```
from utils import *

test_suites = pkl_load('assets/SST2/test_suites.pkl')
INV_test_suites = pkl_load('assets/SST2/INV_test_suites.pkl')
SIB_test_suites = pkl_load('assets/SST2/SIB_test_suites.pkl')

n = 3
df_orig = pd.DataFrame.from_dict(test_suites[0]).head(n)
df_INV  = pd.DataFrame.from_dict(INV_test_suites[0]).head(n)
df_SIB  = pd.DataFrame.from_dict({'data': SIB_test_suites[0]['data'], 
                                  'target': SIB_test_suites[0]['target'],
                                  'ts': SIB_test_suites[0]['ts']}).head(n)

df_orig.rename(columns={'data': 'original'}, inplace=True)
df_INV.rename(columns={'data': 'INV_transformed', 'ts' : 'transforms_applied'}, inplace=True)
df_SIB.rename(columns={'data': 'SIB_transformed', 'ts' : 'transforms_applied'}, inplace=True)

df = pd.concat([df_orig, df_INV, df_SIB], axis=1)

df
```

|**#**|**original**|**target**|**INV\_transformed**|**target**|**transforms\_applied**|**SIB\_transformed**|**target**|**transforms\_applied**|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|0|boisterous and utterly charming |1|boisterous robust+ious and utterly charming|1|['RandomInsertion', 'RandomCharInsert']|boisterous and utterly charming  That being said, I loved it. 💁🏽‍♂|1|['InsertPositivePhrase', 'AddPositiveEmoji']|
|1|pathos-filled but ultimately life-affirming finale |1|рɑthos-fіlled but սⅼtimatеly li𝚏e-/ffirmiոɡ finɑlе |1|['RandomCharSubst', 'HomoglyphSwap']|pathos-filled but ultimately life-affirming finale  https://www.dictionary.com/browse/clunky 🙋|1|['AddNegativeLink', 'AddPositiveEmoji']|
|2|with a lower i.q. than when i had entered |0|with a gloomy i.q. than when i had immerse |0|['ChangeSynonym', 'ChangeHyponym']|with a lower i.q. than when i had entered  👨‍❤‍💋‍👨 That being said, I liked it.|1|['AddPositiveEmoji', 'InsertPositivePhrase']|


## `colab` notebooks

Since our local machines were not even close to being powerful enough to handle the computation requirements of our project, we also created some google `colab` notebooks to help parrallelize our experimental evaluations. 

train_SST2
https://colab.research.google.com/drive/13Gk_hDTJ25s_BPXmH6wmySIM4EI-YsEO?usp=sharing

train_AG_NEWS
https://colab.research.google.com/drive/1vA8K6VX99Zmcr00nk-0-ZmITSLmWfyI-?usp=sharing
