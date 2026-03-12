import os
import re

TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TABLE.md")

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
    Updates the TABLE.md file with the given metrics.
    If only miou is provided (and no dataset), updates the mIoU column for the model/method.
    If dataset, auprc, and fpr95 are provided, updates the respective AuPRC and FPR95 columns.
    """
    if not os.path.exists(TABLE_PATH):
        print(f"Error: {TABLE_PATH} not found.")
        return

    with open(TABLE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the block for the specific Model and Method
    # ERFNET block starts with <td rowspan="3">ERFNET</td>
    # EoMT block starts with <td rowspan="4">EoMT</td>

    # Regex to find the row corresponding to the method inside the model block
    # We will split the HTML by <tr> to process row by row
    rows = content.split('<tr>')
    new_rows = []
    
    current_model = None
    
    for row in rows:
        if '<td rowspan="3">ERFNET</td>' in row:
            current_model = 'ERFNET'
        elif '<td rowspan="4">EoMT</td>' in row:
            current_model = 'EoMT'
            
        is_target_row = False
        
        # Check if this row is for our target model and method
        if current_model == model.upper() or current_model == model:
            # Check if this row is for our target method
            method_patterns = [
                f'<td>{method}</td>',
                f'<td>{method.upper()}</td>',
                f'<td>{method.capitalize()}</td>',
                f'<td>Max Entropy</td>' if method.lower() == 'maxentropy' else 'NON_MATCH'
            ]
            
            if any(p in row for p in method_patterns):
                is_target_row = True
                
        if is_target_row:
            # We found the row, now parse the <td> elements
            td_parts = row.split('<td>')
            
            # td_parts structure depends on whether it's the first row of a model block or not
            # First row has: <td rowspan="N">Model</td><td>Method</td><td>mIoU</td><td>A1</td><td>F1</td>...
            # Other rows have: <td>Method</td><td>mIoU</td><td>A1</td><td>F1</td>...
            
            # Find the mIoU index
            miou_idx = 0
            for i, part in enumerate(td_parts):
                if method.lower() in part.lower() or 'max entropy' in part.lower():
                    miou_idx = i + 1
                    break
                    
            if miou is not None and miou != '-':
                # Update mIoU
                if miou_idx < len(td_parts):
                    # Replace whatever is inside the <td> tag before the </td>
                    td_parts[miou_idx] = re.sub(r'(.*?)(</td>.*)', fr'{miou}\2', td_parts[miou_idx], flags=re.DOTALL)
                    
            if dataset and auprc and fpr95 and dataset in DATASET_TO_COL_INDEX:
                # Update AuPRC and FPR95
                idx_offset = DATASET_TO_COL_INDEX[dataset] * 2
                auprc_idx = miou_idx + 1 + idx_offset
                fpr95_idx = auprc_idx + 1
                
                if auprc_idx < len(td_parts) and fpr95_idx < len(td_parts):
                    td_parts[auprc_idx] = re.sub(r'(.*?)(</td>.*)', fr'{auprc}\2', td_parts[auprc_idx], flags=re.DOTALL)
                    td_parts[fpr95_idx] = re.sub(r'(.*?)(</td>.*)', fr'{fpr95}\2', td_parts[fpr95_idx], flags=re.DOTALL)
                    
            row = '<td>'.join(td_parts)
            
        new_rows.append(row)
        
    new_content = '<tr>'.join(new_rows)
    
    with open(TABLE_PATH, 'w', encoding='utf-8') as f:
        f.write(new_content)
