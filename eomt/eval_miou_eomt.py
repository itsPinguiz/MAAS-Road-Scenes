import sys
import os
# Append the eval path to import existing dataloader structures without duplication
sys.path.append(os.path.abspath('../eval'))

import torch
import time
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
from torchmetrics import JaccardIndex

from dataset import cityscapes
from transform import Relabel, ToLabel
from evalAnomaly_eomt import get_dense_logits

def load_eomt_model(ckpt_path):
    import torch
    from models.vit import ViT
    from models.eomt import EoMT
    from training.mask_classification_semantic import MaskClassificationSemantic

    print(f"Instantiating EoMT with parameters from checkpoint...")
    
    # Parametri esatti estratti dal file .ckpt
    img_size = (1024, 1024)
    num_classes = 19
    
    encoder = ViT(
        img_size=img_size, 
        patch_size=16, 
        backbone_name="vit_base_patch14_reg4_dinov2"
    )

    network = EoMT(
        num_q=100,
        encoder=encoder,
        num_blocks=3,
        masked_attn_enabled=True,
        num_classes=num_classes,
    )

    model = MaskClassificationSemantic(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        attn_mask_annealing_enabled=True,
    ).eval()

    # Caricamento dei pesi
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        
        # Pulizia delle chiavi di PyTorch Lightning se necessario (rimuove "network." se il modello base è già network)
        # Ma nel nostro caso, MaskClassificationSemantic ha un attributo self.network, quindi le chiavi dovrebbero combaciare!
        
        # Mettiamo strict=True. Se fallisce ora, significa che abbiamo ancora un mismatch, ma con questi parametri non dovrebbe!
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Error loading weights: {e}")
        # Fallback senza strict nel caso ci siano chiavi extra non importanti, ma avvisiamo l'utente
        print("Attempting fallback with strict=False...")
        model.load_state_dict(state_dict, strict=False)

    return model

# 19 cityscapes classes + 1 ignored class mapped to 19
NUM_CLASSES = 20
IGNORE_INDEX = 19

def main():
    parser = ArgumentParser()
    parser.add_argument('--datadir', default="../")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--ckpt_path', default="../trained_models/epoch_106-step_19902_eomt.ckpt")
    parser.add_argument('--device', default='cuda:0', help='Device to use for computation')
    args = parser.parse_args()

    model = load_eomt_model(args.ckpt_path)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    import os
    import numpy as np
    from PIL import Image
    from torchmetrics import JaccardIndex
    from tqdm import tqdm

    # Ricerca infallibile con os.walk
    image_paths = []
    for root, dirs, files in os.walk(args.datadir):
        # Controlliamo che sia dentro una cartella 'val' per non valutare il train set
        if 'val' in root.split(os.sep): 
            for file in files:
                if file.endswith("leftImg8bit.png"):
                    image_paths.append(os.path.join(root, file))

    if len(image_paths) == 0:
        print(f"ERRORE CRITICO: Nessuna immagine trovata in {args.datadir}")
        return

    # Metrica: 19 classi (ignorando l'indice 255)
    metric = JaccardIndex(task="multiclass", num_classes=19, ignore_index=255).to(device)

    valid_images_count = 0

    # Mappatura standard di Cityscapes da LabelIds (0-33) a TrainIds (0-18).
    # Tutto ciò che non è in questo dizionario viene mappato a 255 (Background/Ignored)
    mapping_256 = np.ones(256, dtype=np.uint8) * 255
    cityscapes_mapping = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
        25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
    }
    for k, v in cityscapes_mapping.items():
        mapping_256[k] = v

    for img_path in tqdm(image_paths, desc="Evaluating images"):
        # Ora cerchiamo il suffisso corretto che hai mostrato nel terminale: '_labelIds.png'
        gt_path = img_path.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest') \
                          .replace('leftImg8bit', 'gtFine') \
                          .replace('.png', '_labelIds.png')
        
        if not os.path.exists(gt_path):
            print(f"ATTENZIONE: Manca la label per {img_path}\\nCercata in: {gt_path}")
            continue

        valid_images_count += 1

        # Carica Immagine e Label Raw
        img_np = np.array(Image.open(img_path).convert('RGB'))
        label_raw_np = np.array(Image.open(gt_path))
        
        # Mappa i 34 ID raw ai 19 Train ID in modo istantaneo
        label_mapped_np = mapping_256[label_raw_np]
        
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        label_tensor = torch.from_numpy(label_mapped_np).unsqueeze(0).to(device)

        with torch.no_grad():
            dense_logits, _, _ = get_dense_logits(model, img_tensor)
            preds = torch.argmax(dense_logits, dim=1)
            metric.update(preds, label_tensor)
            
        # Alla fine del ciclo for, dopo metric.update()
        del dense_logits, preds, img_tensor, label_tensor
        torch.cuda.empty_cache()

    # Calcola il risultato finale
    mIoU = metric.compute().item()
    print("=======================================")
    print(f"EoMT_mIoU_FINAL: {mIoU * 100:.2f}%")

    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from update_table import update_table_entry
    
    miou_str = f"{mIoU * 100:.2f}"
    # Aggiorniamo la colonna mIoU per tutti i metodi di EoMT (incluso RbA)
    for method in ['MSP', 'MaxLogit', 'Max Entropy', 'RbA']:
        update_table_entry(model="EoMT", method=method, miou=miou_str)
    
    print("Tabella TABLE.md aggiornata con successo con la mIoU di EoMT!")

if __name__ == '__main__':
    main()
