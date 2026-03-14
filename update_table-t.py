import os
import re

TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TABLE_T.md")

DATASET_TO_COL_INDEX = {
    'RoadAnomaly21': 0,
    'RoadObsticle21': 1,
    'Fishyscapes Lost & Found': 2,
    'Fishyscapes Static': 3,
    'RoadAnomaly': 4
}

def update_table_t_entry(model: str, method: str, dataset: str, auprc: str, fpr95: str):
    """
    Updates TABLE_T.md. 
    method string should match exactly the <td> text, e.g., 'MSP (t = 0.5)'
    """
    if not os.path.exists(TABLE_PATH):
        print(f"Error: {TABLE_PATH} not found.")
        return

    with open(TABLE_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    rows = content.split('<tr>')
    new_rows = []
    current_model = None
    
    for row in rows:
        if f'rowspan="4">{model.upper()}</td>' in row:
            current_model = model.upper()
        elif 'rowspan' in row and '</td>' in row:
            # Se troviamo un rowspan ma non è il nostro modello, resettiamo
            current_model = None
            
        is_target_row = False
        if current_model == model.upper() and f'<td>{method}</td>' in row:
            is_target_row = True
                
        if is_target_row:
            td_parts = row.split('<td>')
            # In TABLE_T, l'indice della mIoU è dopo il metodo, poi partono le coppie AuPRC/FPR95
            # Cerchiamo l'indice del metodo per orientarci
            m_idx = 0
            for i, p in enumerate(td_parts):
                if method in p:
                    m_idx = i
                    break
            
            # Offset: m_idx + 1 (mIoU) + 1 (primo AuPRC)
            miou_col_offset = 1 
            idx_offset = DATASET_TO_COL_INDEX[dataset] * 2
            auprc_idx = m_idx + miou_col_offset + 1 + idx_offset
            fpr95_idx = auprc_idx + 1
            
            if auprc_idx < len(td_parts) and fpr95_idx < len(td_parts):
                td_parts[auprc_idx] = re.sub(r'(.*?)(</td>.*)', fr'{auprc}\2', td_parts[auprc_idx], flags=re.DOTALL)
                td_parts[fpr95_idx] = re.sub(r'(.*?)(</td>.*)', fr'{fpr95}\2', td_parts[fpr95_idx], flags=re.DOTALL)
            
            row = '<td>'.join(td_parts)
            
        new_rows.append(row)
        
    with open(TABLE_PATH, 'w', encoding='utf-8') as f:
        f.write('<tr>'.join(new_rows))