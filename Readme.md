# Personality recognition

## Data
myPersonality is a sample of personality scores and Facebook profile data collected by David Stillwell and Michal Kosinski through a Facebook application that implements the NEO-PI-R (five factor personality test, Costa & McCrae) and other psychological tests. This dataset is a subset containing 250 users and about 9900 status updates. It includes Facebook statuses in raw text, gold standard personality labels both as scores and as classes derived froma median split of the scores. It also includes several social network measures such as: network size, betweenness centrality, density, brokerage, transitivity.

The status updates in myPersonality have been anonymized manually, with each proper name replaced by a fixed string (\*PROPNAME\*), though famous named entities such as Chopin or New York have not been replaced.

The data has been split into 80% train and 20% test. You will receive the training subset. The class distributions of the training data are shown in train_splits.png; the test data is also imbalanced but may have different distributions.

## Classification

personality_classifier.py will load the data, run the feature extractor with the supplied configuration file, and train a model to predict each personality trait's binary label  with the extracted features. 

You can also run personality_classifier.py with the --load option to have it load the binary model files stored in the train mode and run them on new data. Execute personality_classifier.py without arguments for usage.

## Submission

Each team will have three tries to submit an experiment configuration including five pickled models (one for each trait), a feature extractor, and a conf file; the configuration will be run on the unseen test data. You can also submit a modified personality_classifier.py if you would like to change the machine learning parameters.

## Evaluation

Evaluation just looks at percent accuracy. More important than accuracy is the thoughtful use of motivated and interesting features.

## Individual component

Each team member should also submit a discussion of the features used and discarded, results, and comparison with papers discussed in class, due via email (proflevitan@gmail.com) next week Wednesday.

