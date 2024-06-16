# Détection de Panneaux de Signalisation avec méthodes de classifications classiques (apprentissage supervisé)

Cette sous partie du projet vise à développer un système de détection de panneaux de signalisation à l'aide de méthodes de classifications classiques (apprentissage supervisé). Le système est capable de détecter et classifier différents types de panneaux et feux de signalisation dans des images.

## Structure du Projet

- `machine_learning/` : Répertoire principal du projet contenant le code source pour l'apprentissage du modèle et la détection des panneaux.
    - `classifiers/` : Répertoire comprenant une liste de notebook de tests de différents classifiers ainsi qu'un fichier concaténant les résultats
    - `main.py` : Point d'entrée pour lancer le programme.
    - `config.py` : Fichier de configuration.

- `data/` : Répertoire pour stocker les données d'entraînement et de test.
- `utils/` : Répertoire contenant des utilitaires et des scripts auxiliaires.

## Utilisation

Pour exécuter le programme et lancer la détection de panneaux de signalisation, utilisez la commande suivante depuis la racine du projet :

```bash
python -m machine_learning.main
