# AS_project-Skim-RNN

Implémentation et tests de l'article proposé (et accepté) à l'ICLR : Neural Speed Reading via SKIM-RNN
https://openreview.net/pdf?id=Sy-dQG-Rb

## Idéé Principale

Un modèle de RNN en seq2seq qui effectue, pour chaque séquence en cours, un tirage aléatoire sur une distribution (à apprendre). Ce tirage indique si le mot est à lire en entier, dans quel cas il passe dans un RNN à grande dimension latente, ou à 'skimmer', dans quel cas il passe dans un RNN à plus faible dimensions.

Cela engendre peu de pertes en précision (voire une légère amélioration), et un gain computationnel important sur CPU. 

## Fichiers

Le projet comporte
- un module de preprocessing pour créer et sauvegarder des corpus de textes valides
- un fichier outils avec différentes fonctions utilisées dans le projet
- un fichier modèles qui contient les implémentations du sélecteur, des RNN, et enfin du skim-RNN avec pyTorch
- un fichier train.py pour l'apprentissage du skim-RNN, et un fichier pour sa validation, quelques tests et scripts

## Expériences réalisées

Sur les données Rotten Tomatoes, apprentissage de différents modèles et dimensions de Skim RNN, et benchmark avec les baselines LSTM Jump et RNN classique. 
Etude de l'impact du coefficient de régularisation pour favoriser le skimrate, et visualisations des résultats sur des reviews rotten tomatoes.
