# Détection de Panneaux de Signalisation avec Réseaux de Convolution (CNN)

Cette sous partie du projet vise à développer un système de détection de panneaux de signalisation à l'aide de réseaux de neurones convolutionnels (CNN). Le système est capable de détecter et classifier différents types de panneaux et feux de signalisation dans des images.

## Structure du Projet

- `deep_learning/` : Répertoire principal du projet contenant le code source pour l'apprentissage du modèle et la détection des panneaux.
    - `main.py` : Point d'entrée pour lancer le programme.
    - `signClassifier.py` : Module contenant les définitions des modèles CNNs pour la classification et détection des panneaux/feux de signalisation.
    - `config.py` : Fichier de configuration du réseau de neurones.
    - `utiles.py` : Module contenant les fonctions utilitaires du module.

- `data/` : Répertoire pour stocker les données d'entraînement et de test.
- `utils/` : Répertoire contenant des utilitaires et des scripts auxiliaires.

## Utilisation

Pour exécuter le programme et lancer la détection de panneaux de signalisation, utilisez la commande suivante depuis la racine du projet :

```bash
python -m deep_learning.main
