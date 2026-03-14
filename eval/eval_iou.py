import os
import torch
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
import numpy as np

from erfnet import ERFNet
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry

NUM_CLASSES = 20
IGNORE_INDEX = 19

input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, IGNORE_INDEX),   # ignore label to 19
])

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
            else:
                continue
        else:
            own_state[name].copy_(param)
    return model

def main(args):
    modelpath = os.path.join(args.loadDir, args.loadModel)
    weightspath = os.path.join(args.loadDir, args.loadWeights)
    
    model = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    model.eval()

    # Ricerca infallibile con os.walk (come fatto in EoMT)
    image_paths = []
    for root, dirs, files in os.walk(args.datadir):
        if 'val' in root.split(os.sep): 
            for file in files:
                if file.endswith("leftImg8bit.png"):
                    image_paths.append(os.path.join(root, file))

    if len(image_paths) == 0:
        print(f"ERRORE CRITICO: Nessuna immagine trovata in {args.datadir}")
        return

    iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=IGNORE_INDEX)
    
    # 1. Definiamo la mappatura da labelIds (0-33) a trainIds (0-18)
    mapping_256 = np.ones(256, dtype=np.uint8) * 255
    cityscapes_mapping = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
        19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
        25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
    }
    for k, v in cityscapes_mapping.items():
        mapping_256[k] = v

    iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=IGNORE_INDEX)

    for img_path in tqdm(image_paths, desc="Evaluating ERFNet"):
        gt_path = img_path.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest') \
                          .replace('leftImg8bit', 'gtFine') \
                          .replace('.png', '_labelIds.png')
        
        if not os.path.exists(gt_path):
            continue

        # 2. Carichiamo le immagini
        img = Image.open(img_path).convert('RGB')
        
        # 3. Carichiamo la maschera RAW e applichiamo la mappatura prima di trasformarla
        label_raw_np = np.array(Image.open(gt_path))
        label_mapped_np = mapping_256[label_raw_np]
        gt = Image.fromarray(label_mapped_np) # Riconvertiamo in PIL Image per i transform

        if not args.cpu:
            img_t = input_transform_cityscapes(img).unsqueeze(0).cuda()
            gt_t = target_transform_cityscapes(gt).unsqueeze(0).cuda()
        else:
            img_t = input_transform_cityscapes(img).unsqueeze(0)
            gt_t = target_transform_cityscapes(gt).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_t)
            
        pred = outputs.max(1)[1].unsqueeze(1).data
        
        # Ora i dati sono corretti alla radice, ma manteniamo la sicurezza
        gt_t[gt_t >= NUM_CLASSES] = IGNORE_INDEX
        pred[pred >= NUM_CLASSES] = IGNORE_INDEX
        
        iouEvalVal.addBatch(pred, gt_t)

    iouVal, iou_classes = iouEvalVal.getIoU()

    print("=======================================")
    print(f"ERFNet_mIoU_FINAL: {iouVal.item() * 100:.2f}%")

    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from update_table import update_table_entry
    
    miou_str = f"{iouVal.item() * 100:.2f}"
    # Aggiorniamo la colonna mIoU per tutti i metodi di ERFNet
    for method in ['MSP', 'MaxLogit', 'Max Entropy']:
        update_table_entry(model="ERFNET", method=method, miou=miou_str)
    
    print("Tabella TABLE.md aggiornata con successo con la mIoU di ERFNet!")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--datadir', default="../Datasets/Cityscapes")
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())