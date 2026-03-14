import os
import torch
import torch.nn.functional as F
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

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

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully")
    model.eval()

    # Ricerca infallibile con os.walk (come fatto in EoMT)
    image_paths = []
    for root, dirs, files in os.walk(args.datadir):
        if 'val' in root.split(os.sep): 
            for file in files:
                if file.endswith("leftImg8bit.png"):
                    image_paths.append(os.path.join(root, file))

    print(f"DEBUG: Trovate {len(image_paths)} immagini 'leftImg8bit.png' nella cartella 'val'.")

    if len(image_paths) == 0:
        print(f"ERRORE CRITICO: Nessuna immagine trovata in {args.datadir}")
        return

    iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=IGNORE_INDEX)

    for img_path in tqdm(image_paths, desc="Evaluating ERFNet"):
        # Ricostruzione sicura del path della ground truth
        gt_path = img_path.replace('leftImg8bit_trainvaltest', 'gtFine_trainvaltest') \
                          .replace('leftImg8bit', 'gtFine') \
                          .replace('.png', '_labelIds.png')
        
        if not os.path.exists(gt_path):
            print(f"ATTENZIONE: Manca la label per {img_path}")
            continue

        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path)

        if not args.cpu:
            img_t = input_transform_cityscapes(img).unsqueeze(0).cuda()
            gt_t = target_transform_cityscapes(gt).unsqueeze(0).cuda()
        else:
            img_t = input_transform_cityscapes(img).unsqueeze(0)
            gt_t = target_transform_cityscapes(gt).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_t)
            
        # 1. Estraiamo la predizione (indice della classe con probabilità più alta)
        pred = outputs.max(1)[1].unsqueeze(1).data
        
        # 2. FIX CUDA: Assicuriamoci che nessun valore superi NUM_CLASSES - 1 (19)
        # Sostituiamo eventuali valori strani (come 255) con l'indice di ignoranza (IGNORE_INDEX)
        gt_t[gt_t >= NUM_CLASSES] = IGNORE_INDEX
        pred[pred >= NUM_CLASSES] = IGNORE_INDEX
        
        # 3. Ora possiamo passare i tensori sicuri alla funzione di valutazione
        iouEvalVal.addBatch(pred, gt_t)

    iouVal, iou_classes = iouEvalVal.getIoU()

    print("=======================================")
    print(f"ERFNet_mIoU_FINAL: {iouVal.item() * 100:.2f}%")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--datadir', default="../Datasets/Cityscapes")
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())