# Détection de Panneaux de Signalisation avec méthodes de classifications classiques (apprentissage supervisé)

Cette sous partie du projet vise à développer un système de détection de panneaux de signalisation à l'aide de méthodes de classifications classiques (apprentissage supervisé). Le système est capable de détecter et classifier différents types de panneaux et feux de signalisation dans des images.

## Structure du Projet

- `supervised_learning/` : Répertoire principal du projet contenant le code source pour l'apprentissage du modèle et la détection des panneaux.
    - `classifiers/` : Répertoire comprenant les informations de différents classifieurs
        - `saves/` : Répertoire de sauvegarde des meilleures classifieurs
    - `main.py` : Point d'entrée pour lancer le programme.
    - `config.py` : Fichier de configuration.
    - `utils.py` : Fichier contenant des scripts utiles aux processus d'apprentissages.

- `data/` : Répertoire pour stocker les données d'entraînement et de test.
- `utils/` : Répertoire contenant des utilitaires et des scripts auxiliaires.

## Utilisation

Pour créer, entrainer et tester les différents classifieurs binaire, lancer la commande suivante :

```bash
python -m supervised_learning.create_classifiers
```

Pour exécuter le programme et lancer la détection de panneaux de signalisation, utilisez la commande suivante depuis la racine du projet :

```bash
python -m supervised_learning.detection_selective_search INPUT_IMAGES_FOLDER OUTPUT_LABEL_FOLDER
```

