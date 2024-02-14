# P7_dashboard_bokeh
Contexte du projet 

Il est réalisé dans le cadre du parcours diplômant de Data Scientist d'Openclassrooms (projet n°7)

Nature du projet

Nous avons un organisme de crédit qui s'appelle *Prêt à dépenser* et qui propose des crédits à la consommation pour des clients ayant peu ou pas du tout d'historique de prêt.

## **Implémentation d'un modèle de scoring :** 

L’entreprise nous sollicite en tant que data scientist afin de mettre en place un outil de scoring crédit qui calcule la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. 

**L'ensemble de l'analyse et de la modélisation est disponible via des "Notebook Jupyter" dans le dossier "Notebook" de ce repository.**

Les données originales sont téléchargeables sur Kaggle à [cette adresse](https://www.kaggle.com/c/home-credit-default-risk/data)


## **Dasboard interactif réalisé avec Bokeh :**

De plus, les chargés de relation client souhaite pouvoir expliquer les motifs d'octroi ou de refus de crédit. Le souci de transparence vis-à-vis des clients fait parti des valeurs de l'entreprise. Le dashboard interactif permettra donc aux chargés de relation client d'expliquer de la manière la plus transparente possible les décisions d’octroi de crédit.

*__Spécifications du dashboard__* : Il contient les fonctionnalités suivantes :

- Il permet de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
- Il permet de visualiser des informations descriptives relatives à un client (via un système de filtre).
- Il permet de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.
