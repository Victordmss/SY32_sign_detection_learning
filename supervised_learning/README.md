# Détection de Panneaux de Signalisation avec méthodes de classifications classiques (apprentissage supervisé)

Cette sous partie du projet vise à développer un système de détection de panneaux de signalisation à l'aide de méthodes de classifications classiques (apprentissage supervisé). Le système est capable de détecter et classifier différents types de panneaux et feux de signalisation dans des images.

## Structure du Projet

- `supervised_learning/` : Répertoire principal du projet contenant le code source pour l'apprentissage du modèle et la détection des panneaux.
    - `classifiers/` : Répertoire comprenant les informations de différents classifieurs (notebooks de tests et classifieurs sauvegardés)
        - `saves/` : Répertoire de sauvegarde des meilleures classifieurs
    - `main.py` : Point d'entrée pour lancer le programme.
    - `config.py` : Fichier de configuration.
    - `utils.py` : Fichier contenant les fonctions utilitaires du module.
    - `create_classifiers.py` : Fichier permettant la création de classifieurs SVM binaires et multiclasses utilisés par les modules de détection.
    - `detection_selective_search.py` : Fichier contenant le système de détection par recherche sélective.
    - `detection.py` : Fichier contenant le système de détection par fenêtres glissantes dynamiques successives.
    - `retrain_classifiers.py` : Fichier permettant le réentrainement des classifieurs sur des données de détection par extraction des faux positifs.
    - `vue.py` : Fichier contenant un module de visualisation de données de résultats aléatoire (code de base extrait du cours)

- `data/` : Répertoire pour stocker les données d'entraînement et de test.
- `utils/` : Répertoire contenant des utilitaires et des scripts auxiliaires.

## Utilisation

Pour créer, entrainer et tester les différents classifieurs binaire, lancer la commande suivante :

```bash
python -m supervised_learning.create_classifiers
```

Pour exécuter le programme de détection par fenêtres glissantes dynamiques successives :

```bash
python -m supervised_learning.detection_selective_search INPUT_IMAGES_FOLDER OUTPUT_CSV_FILE
```

Pour exécuter le programme de détection par recherche sélective :

```bash
python -m supervised_learning.detection_selective_search INPUT_IMAGES_FOLDER OUTPUT_CSV_FILE
```

Pour exécuter le programme de réentrainement des classifieurs sur les résultats (dataset TRAIN)

```bash
python -m supervised_learning.retrain_classifieurs RESULTS_CSV_FILE
```

