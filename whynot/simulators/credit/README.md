# Credit Simulator

This credit simulator reprises on a model of strategic classification in which
an institution classifies the creditworthiness of loan applicants, and agents
react to the institution’s classifier by manipulating their features to increase
the likelihood that they receive a favorable classification. The underlying
data comes from a Kaggle credit scoring dataset, and the classification dynamics
were used in the following paper:

    Perdomo, Juan C., Tijana Zrnic, Celestine Mendler-Dünner, and Moritz Hardt.
    "Performative Prediction." arXiv preprint arXiv:2002.06673 (2020).

The model was originally used to qualitatively analyze the long-run properties
of repeated retraining of classifiers in the face of strategic adaptation.
