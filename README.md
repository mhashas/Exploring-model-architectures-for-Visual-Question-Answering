# Exploring model architectures for Visual Question Answering
This repository contains the corresponding training code for [**the project**](https://www.overleaf.com/read/czcchvjrjjgz).

We address the problem of Visual Question Answering, which requires both image and language understanding to answer a question about a given photograph. We describe 2 models for this task: a simple bag-of-words baseline and an improved Long Short Term Memory-based approach.

### Bag of Words model
<p align="center">
  <img src="images/bow_visual.png">
</p>

### LSTM model
<p align="center">
  <img src="images/lstm2.png">
</p>

## Training 
1. Download the [data](https://github.com/timbmg/NLP1-2017-VQA) and place it in the [data folder](data/).
2. Check the [available parser options](application.py).
4. Train the networks using the provided [file](application.py):
`python application.py --model_type bow|lstm`
## Results 

### Both models are correct

<p align="center">
  <img src="images/both-correct.png">
</p>


### Both models are wrong
<p align="center">
  <img src="images/both-wrong.png">
</p>

### Only BoW is correct
<p align="center">
  <img src="images/bow-correct.PNG">
</p>

### Only LSTM is correct
<p align="center">
  <img src="images/lstm-correct.PNG">
</p>

