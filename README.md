# Sibylvariant Transformations for Robust Text Classification

## Summary
The vast majority of text transformation techniques in NLP are inherently limited in their ability to expand input space coverage due to an implicit constraint to preserve the original class label. In this work, we propose the notion of sibylvariance (SIB) to describe the broader set of transforms that relax the label-preserving constraint, knowably vary the expected class, and lead to significantly more diverse input distributions. We  offer a unified framework to organize all data transformations, including two types of SIB: (1) Transmutationsconvert one discrete kind into another, (2) Mixture Mutations blend two or more classes together. To explore the role of sibylvariance within NLP, we implemented 41 text transformations, including several novel techniques like Concept2Sentence and SentMix. Sibylvariance also enables a unique form of adaptive training that generates new input mixtures for the most confused class pairs, challenging the learner to differentiate with greater nuance. Our experiments on six benchmark datasets strongly support the efficacy of sibylvariance for generalization performance, defect detection, and adversarial robustness.

## Team 
This project is developed by Professor [Miryung Kim](http://web.cs.ucla.edu/~miryung/)'s Software Engineering and Analysis Laboratory at UCLA. 
If you encounter any problems, please open an issue or feel free to contact us:

- [Fabrice Harel-Canada](https://fabrice.harel-canada.com/): PhD student, fabricehc@cs.ucla.edu;
- [Muhammad Ali Gulzar](https://people.cs.vt.edu/~gulzar/): Assistant Professor, gulzar@cs.vt.edu;
- [Nanyun Peng](https://vnpeng.net/): Assistant Professor, violetpeng@cs.ucla.edu;
- [Miryung Kim](https://web.cs.ucla.edu/~miryung/): Professor, miryung@cs.ucla.edu;

## How to cite 
Please refer to our Findings of ACL'22 paper, Sibylvariant Transformations for Robust Text Classification for more details. 

### Bibtex  
```
[PENDING]
```

DOI Link [PENDING]

# `Sibyl`

`Sibyl` is a tool for generating new data from what you have already. Thus far, we have implemented over 40 different text transforms and a handful of image transforms.

There are two primary kinds of transformations:
- Invariant (INV) : transform the input, but the expected output remains the same.
  - ex. "I love NY" + `Emojify` = "I 💗 NY", which has an INV effect on the sentiment.
- Sibylvariant (SIB) : transform the input and the expected output may change in some way.
  - ex. "I love NY" + `ChangeAntonym` = "I hate NY", which has a SIB effect on the sentiment (label inverted from positive (1) to negative (0)).
  
Some transformations also result in *soft labels*, called sibylvariant mixture mutations. For example, within topic classification you could use `SentMix` on your data to randomly combine two inputs from different classes into one input and then shuffle the sentences around. This new data would have a new label with probabilities weighted according to the relative length of text contributed by the two inputs. Illustrating with `AG_NEWS`, if you mix an article about sports with one about business, it might result in a new soft label like [0, 0.5, 0.5, 0]. The intuition here is that humans would be able to recognize that there are two topics in a document and we should expect our models to behave similarly (i.e. the model predictions should be very close to 50/50 on the expected topics). 

## Installation

The `Sibyl` tool can be installed locally or via pypi:

```
pip install sibyl-tool
```

This repository contains a snapshot of the `Sibyl` tool for a particular point in time, roughly corresponding to the paper submission. Any future updates to the tool will be made under this [repo](https://github.com/fabriceyhc/Sibyl). 

## Examples

### Concept2Sentence

This transformation intakes a sentence, its associated integer label, and (optionally) a dataset name that is supported by huggingface/datasets. It works by extracting keyword concepts from the original sentence, passing them into a BART transformer trained to generate a new, related sentence which reflects the extracted concepts. Providing a dataset allows the function to use transformers-interpret to identify the most critical concepts for use in the generative step.

```
transform = Concept2Sentence(dataset='sst2')

in_text = "a disappointment for those who love alternate versions of the bard, particularly ones that involve deep fryers and hamburgers."
in_target = 0 # (sst2 dataset 0=negative, 1=positive)

out_text = transform(in_text, in_target)
```

```
Extracted Concepts: ['disappointment']
New Sentence: "disappointment is a common sight among many people."
```

*NOTE: We highly recommend having a gpu for generative transforms like this one.*


### SentMix

This transformation takes in a set of inputs and targets, blends the sentences together and generates a new target label that reflects the ratio of words contributed from each of the source sentences. It produces a soft label that reflects mixed class membership and forces the model to differentiate classes with greater degrees of nuance. 

```
transform = SentMix()

in_text = ["The characters are unlikeable and the script is awful. It's a waste of the talents of Deneuve and Auteuil.", 
           "Unwatchable. You can't even make it past the first three minutes. And this is coming from a huge Adam Sandler fan!!1",
           "An unfunny, unworthy picture which is an undeserving end to Peter Sellers' career. It is a pity this movie was ever made.",
           "I think it's one of the greatest movies which are ever made, and I've seen many... The book is better, but it's still a very good movie!",
           "The only thing serious about this movie is the humor. Well worth the rental price. I'll bet you watch it twice. It's obvious that Sutherland enjoyed his role.",
           "Touching; Well directed autobiography of a talented young director/producer. A love story with Rabin's assassination in the background. Worth seeing"
]

in_target = [[0], [0], [0], [1], [1], [1]] # (imdb dataset 0=negative, 1=positive)

new_text, new_target = transform((in_text, in_target), num_classes=2)
print(new_text[0])
print(new_target[0])
```

```
'The book is better, but it\'s still a very good movie!" It\'s a waste of the talents of Deneuve and Auteuil. I think it\'s one of the greatest movies which are ever made, and I\'ve seen many... b"The characters are unlikeable and the script is awful.',

[0.4380165289256198, 0.5619834710743802]
```

## Evaluation Code

To recreate the evaluations performed for the paper, you can run the scripts provided in the `eval` folder after installing `Sibyl`. First, generate the datasets (to avoid variability from random seed selections), then run each of the `rq_*.py` files. Note that this will take an incredibly long time to process. 