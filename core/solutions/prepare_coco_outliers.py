import os
import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import random
from alive_progress import alive_bar # opzionale, in alternativa log basico
from collections import defaultdict
import logging

# --- CONFIGURAZIONI GLOBALI ---
# Aggiorna questi percorsi in base alla struttura locale del tuo download COCO
IMG_DIR = "/mnt/Shared-Data/Code/github/itsPinguiz/MAAS-Road-Scenes/Datasets/COCO/val2017"
ANN_FILE = "/mnt/Shared-Data/Code/github/itsPinguiz/MAAS-Road-Scenes/Datasets/COCO/annotations_trainval2017/annotations/instances_val2017.json"

# Abbiamo assunto che lo script venga lanciato dalla root (o configuralo assoluto)
OUTPUT_DIR = os.path.join("Datasets", "Outliers_COCO")

# Classi ritenute ideali come "Anomalie Stradali" 
VALID_CLASSES = [
    'dog', 'cat', 'cow', 'sheep', 
    'suitcase', 'frisbee', 'sports ball', 
    'chair', 'couch', 'potted plant',
    'bear', 'horse', 'backpack', 'umbrella', 'trash can' # bonus safe-classes valutabili
]

# Numero massimo di cutout per classe per mantenere bilanciamento e non saturare le CPU
MAX_SAMPLES_PER_CLASS = 100 

# Logger Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    if not os.path.exists(ANN_FILE) or not os.path.exists(IMG_DIR):
        logger.error(f"Errore: i percorsi IMG_DIR o ANN_FILE non sono stati trovati.\nAssicurati di assegnare i path reali in cima a questo script.\nIMG: {IMG_DIR}\nANN: {ANN_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Inizializzazione PyCocoTools in corso...")
    
    # Inizializza l'API COCO (Carica JSON in memoria, può metterci un po')
    coco = COCO(ANN_FILE)
    
    # Crea mappa per gli ID Classe basati solo su quelli che ci interessano
    class_ids = coco.getCatIds(catNms=VALID_CLASSES)
    categories = coco.loadCats(class_ids)
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # Array per raggruppare le annotazioni compatibili (per classe)
    annotations_by_class = defaultdict(list)
    
    # Popoliamo le annotazioni pre-selezionate controllando l'area in canali validi
    for cat_id in class_ids:
        ann_ids = coco.getAnnIds(catIds=cat_id, iscrowd=False) # iscrowd=False ignora segmentazioni vaghe
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            # Filtro rapido per scartare annotazioni corrotte o troppo piccole prima ancora di caricarle
            if ann['area'] > 2500 and len(ann.get('segmentation', [])) > 0:
                annotations_by_class[cat_id].append(ann)
    
    total_extracted = 0
    
    for cat_id, name in cat_id_to_name.items():
        anns_for_class = annotations_by_class[cat_id]
        
        # Campionamento Casuale
        if len(anns_for_class) > MAX_SAMPLES_PER_CLASS:
            anns_for_class = random.sample(anns_for_class, MAX_SAMPLES_PER_CLASS)
            
        logger.info(f"Classe [{name}]: elaborazione di {len(anns_for_class)} istanze...")
        
        for ann in anns_for_class:
            try:
                # Recupera e apri immagine fisica
                img_data = coco.loadImgs(ann['image_id'])[0]
                img_path = os.path.join(IMG_DIR, img_data['file_name'])
                
                if not os.path.exists(img_path):
                    continue
                
                # Apre immagine in RGB convertendola via PIL e numpy
                img_rgb = np.array(Image.open(img_path).convert('RGB'))
                
                # Decodifica segmentazione poligonale in una Mask Numpy (0=bg, 1=fg)
                mask = coco.annToMask(ann)
                
                # Ricava BBOX intero 
                x_bbox, y_bbox, w_bbox, h_bbox = [int(v) for v in ann['bbox']]
                
                # Sanity test estremo: check BBOX limiti per non crashare
                if w_bbox <= 0 or h_bbox <= 0:
                    continue
                    
                x_min = max(0, x_bbox)
                y_min = max(0, y_bbox)
                x_max = min(img_rgb.shape[1], x_bbox + w_bbox)
                y_max = min(img_rgb.shape[0], y_bbox + h_bbox)
                
                # Crop di immagine e mask alla bounding box esatta
                img_crop = img_rgb[y_min:y_max, x_min:x_max]
                mask_crop = mask[y_min:y_max, x_min:x_max]
                
                # Generazione dei 4 canali RGBA
                # 1. Crea uno shallow space RGBA dello stesso shape del crop ma 4 Canali (Alpha)
                # Inizializziamo a zero per trasparenza totale base
                rgba_crop = np.zeros((img_crop.shape[0], img_crop.shape[1], 4), dtype=np.uint8)
                
                # 2. Inseriamo i valori RGB pre-esistenti originali e l'alpha mask a 255 dove = 1 
                # (Sfruttiamo broadcasting booleano veloce Numpy)
                fg_indices = (mask_crop == 1)
                rgba_crop[fg_indices, 0:3] = img_crop[fg_indices]  # Red Green Blue
                rgba_crop[fg_indices, 3] = 255                     # Alpha a 100% visibile 
                
                # Costruisce Image PIL e salva RGBA su disco
                result_img = Image.fromarray(rgba_crop, 'RGBA')
                
                save_path = os.path.join(OUTPUT_DIR, f"{name}_{ann['id']}.png")
                result_img.save(save_path, "PNG")
                total_extracted += 1

            except Exception as e:
                # Gestisce rare eccezioni come immagini danneggiate, memory overflow in poligoni non validi, ecc.
                logger.debug(f"Errore annotazione {ann['id']}: {e}")
                continue

    logger.info(f"Estrazione OOD Completata! Generati: {total_extracted} outlier puri con trasparenza.")
    logger.info(f"Risultati disponibili in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
