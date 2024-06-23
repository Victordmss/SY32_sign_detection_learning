import cv2
import matplotlib.pyplot as plt

def selective_search(image, visualize=True, visulize_count=100):
    # Convert image to BGR format for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Initialiser la recherche sélective
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image_bgr)

    # Utiliser la recherche sélective en mode rapide (ou en mode qualité)
    ss.switchToSelectiveSearchFast()  # Pour la recherche rapide
    # ss.switchToSelectiveSearchQuality()  # Pour une recherche plus précise

    # Obtenir les régions candidates
    roi = ss.process()

    if visualize:
        # Dessiner les régions candidates sur l'image
        for (x, y, w, h) in roi[:visulize_count]:  # Limiter à 100 régions pour la visualisation
            cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convertir l'image BGR en RGB pour Matplotlib
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Afficher l'image avec les régions candidates
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.axis('off')  # Cacher les axes pour une meilleure visualisation
        plt.show()
        
    return roi

img = cv2.imread("./data/val/images/0846.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Assurer que l'image est en RGB
result = selective_search(img_rgb)
