import os
import sys
from bs4 import BeautifulSoup

from core.utility.config_loader import cfg

TABLE_PATH = cfg.paths.tables.main

# Datasets in the order they appear in the HTML table
DATASET_TO_COL_INDEX = {
    'RoadAnomaly21': 0,
    'RoadObsticle21': 1,
    'Fishyscapes Lost & Found': 2,
    'Fishyscapes Static': 3,
    'RoadAnomaly': 4
}

def update_table_entry(model: str, method: str, dataset: str = None, miou: str = None, auprc: str = None, fpr95: str = None):
    """
    Updates the TABLE.md file with the given metrics using BeautifulSoup.
    If only miou is provided, updates the mIoU column for the model/method.
    If dataset, auprc, and fpr95 are provided, updates the respective AuPRC and FPR95 columns.
    """
    if not os.path.exists(TABLE_PATH):
        print(f"Error: {TABLE_PATH} not found.")
        return

    with open(TABLE_PATH, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    tbody = soup.find('tbody')
    if not tbody:
        print("Error: <tbody> not found in TABLE.md.")
        return

    current_model = None
    target_method = method.strip().lower()

    for tr in tbody.find_all('tr', recursive=False):
        tds = tr.find_all('td', recursive=False)
        if not tds:
            continue

        # Check if this row starts a new model block (first td has rowspan)
        first_td = tds[0]
        has_rowspan = first_td.has_attr('rowspan')
        
        if has_rowspan:
            current_model = first_td.get_text(strip=True).upper()
            base_idx = 1
        else:
            base_idx = 0

        if current_model == model.strip().upper():
            if base_idx < len(tds):
                row_method = tds[base_idx].get_text(strip=True).lower()
                
                # Loose matching for "Max Entropy" -> "maxentropy"
                is_match = (row_method == target_method) or (target_method == 'maxentropy' and row_method == 'max entropy')
                
                if is_match:
                    if miou is not None and miou != '-':
                        miou_idx = base_idx + 1
                        if miou_idx < len(tds):
                            tds[miou_idx].string = str(miou)
                    
                    if dataset and auprc and fpr95 and dataset in DATASET_TO_COL_INDEX:
                        ds_col = DATASET_TO_COL_INDEX[dataset]
                        metrics_start_idx = base_idx + 2
                        
                        auprc_idx = metrics_start_idx + (ds_col * 2)
                        fpr95_idx = auprc_idx + 1
                        
                        if auprc_idx < len(tds) and fpr95_idx < len(tds):
                            tds[auprc_idx].string = str(auprc)
                            tds[fpr95_idx].string = str(fpr95)
                            
                    break

    # Save the updated table using str(soup) to preserve formatting
    with open(TABLE_PATH, 'w', encoding='utf-8') as f:
        f.write(str(soup))
