# Projet de Classification et Détection de Panneaux et Feux de Signalisation

## Description

Ce projet a pour objectif de classifier et détecter les panneaux et feux de signalisation en utilisant des méthodes d'apprentissage supervisé et de deep learning. Le projet est organisé en plusieurs répertoires pour faciliter la compréhension et l'utilisation des différentes parties du code.

## Structure du Projet

Le projet est structuré comme suit :

- `supervised_learning/` : Contient les modules relatifs à l'apprentissage supervisé.
- `deep_learning/` : Contient les modules relatifs au deep learning.
- `data/` : Contient les données utilisées pour l'entraînement, le test et la validation. Les datasets sont organisés en trois sous-répertoires :
  - `train/` : Données d'entraînement
  - `test/` : Données de test
  - `val/` : Données de validation
- `utils/` : Contient des fonctions importantes et utiles à tous les modules.
- `docs/` : Contient les documents importants du projet.
  - `Rapport.pdf/` : Rapport au format pdf
  - `Rapport/` : Rapport au format latex
  - `Projet.pdf/` : Consigne du projet


## Utilisation

Chaque répertoire (`supervised_learning` et `deep_learning`) contient un fichier `README.md` qui explique en détail comment utiliser les modules respectifs. Veuillez vous référer à ces fichiers pour des instructions spécifiques sur l'utilisation et l'entraînement des modèles.

## Installation

1. Clonez le dépôt :
   ```bash
   git clone https://gitlab.utc.fr/vdemessa/sy32_project.git
2. Accedez au répertoire du projet :
   ```bash
   cd SY32_sign_detection_learning/
3. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt

## Contributeurs 

- Victor Demessance
- Mathieu Kozdeba

Merci aux étudiants de [SY32](https://vision.uv.utc.fr/doku.php?id=sy32) (Cours "Visions et apprentissage") de l'Université de Technologie de Compiègne pour la création et labelisation du dataset. 
