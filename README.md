# Description
A small project for the SENIOR NLP ENGINEER position at HiringBranch.  

# How to run
## Installation
- Clone the project. `git clone http://github.com/ierezell/hiring_branch`
- Then from the project root :

### With pip 
- Create a venv : `python -m venv .venv`
- Activate it : `source ./.venv/bin/activate`
- install the requirements : `pip install -r requirements.txt`

### With poetry 
- Install poetry (a python project manager) : https://python-poetry.org/docs/master/#installation 
- Install the project from the root with `poetry install`
- Run commands in the project with `poetry run my_command` or activate a shell with `poetry shell` and then `my_command`

- The main entry point is `cli` to run all the the commands. (get help with `cli --help`).

## Developer Flow : 
- Use the cli to split the corpus to passive and active sentences.
- Then train the model or contact me to get pre-trained weights. 
- Optimize the model (or contact me to get onnx weights).
- Finally launch the server
- (Optional) Use `poetry build` to generate a wheel to distribute the package.


# HTTP API 

### / or /docs
Will show the swagger documentation created by fastAPI. You can also use this UI to test the routes with the "Try out" button.

### /inference 
Expect a json body parameter named text with one string.
Returns a json with one field : "logits" containing the classification result (float between 0 and 1). Close to 1 means quite certain of the text to be a passive sentence.

# Testing 
All the tests were made with pyTest. Please refer to their doc for all the options. 

To run all the tests, use : `poetry run pytest .` (or `pytest .`)

# Thoughts 
## Classifier 
- There is a small bias in the data : most of the passive sentences are longer than the non-passive ones. 
- There is also more non-passive sentences but I re-balanced the dataset.
- DataLoader could be improved (more shuffling/sampling/augmentation) to get better results. 
- To train the classification model, I first used bert with a classification head but it's too much parameters for so few data. Freezing the embedding part and training only the head lead to correct results. Once more data is gathered/generated, we could finetune the embedding layer to obtain better performance. 
- For this task in particular, the order of the words are important so any bag of word / non positional encoding seems less efficient (thus the use of bert). I'm afraid that those models would use other feature (like sentence length) to classify but I didn't found the time to test it. 
- I used an "MLP-ish" head as I was already using pytorch lightning but any SVM/Regression or other algo could do. 
- I didn't removed punctuation or lemmatize/stem words as I trust bert embeddings to deal with that. However, with spacy or nltk it would be easy to do so. 

## Server
- Api was made using fastApi for speed (uvloop and uvicorn), type hints and easy testing. 
- The api was made to host the trained classifier but in a production grade environment, for equal performance (+-5%) using the rule base method would be faster/lighter and thus more suitable. 
- The model was compressed using onnx to gain speed and size (""production like""). 
- Deployment option could be : lambda function, aws inferentia or other Ec2 based (gpu or not), with onnx or equivalent compilation/optimization.

## Utils
- For the sentence splitter, the custom rule base one is fine but spacy one (or nltk) would be better. 