import os
from bs4 import BeautifulSoup

from core.utility.config_loader import cfg

TABLE_PATH = cfg.paths.tables.main

# Datasets in the order they appear in the HTML table
DATASET_TO_COL_INDEX = {
    'RoadAnomaly21': 0,
    'RoadObsticle21': 1,
    'Fishyscapes Lost & Found': 2,
    'Fishyscapes Static': 3,
    'RoadAnomaly': 4,
}

# Schema: { model_name: [list of method names] }
# rowspan is derived automatically from len(methods)
_TABLE_SCHEMA = {
    'ERFNet': ['MSP', 'MaxLogit', 'Max Entropy'],
    'EoMT':   ['MSP', 'MaxLogit', 'Max Entropy', 'RbA'],
}

_DATASETS_ORDER = list(DATASET_TO_COL_INDEX.keys())


def _empty_td(soup, text='-'):
    td = soup.new_tag('td')
    td.string = text
    return td


def _build_empty_table() -> str:
    """Build a blank TABLE.md HTML string with '-' placeholders."""
    soup = BeautifulSoup('', 'html.parser')

    table = soup.new_tag('table')

    # --- thead ---
    thead = soup.new_tag('thead')
    # Row 1: dataset group headers
    tr1 = soup.new_tag('tr')
    for _ in range(3):          # Model / Method / mIoU
        tr1.append(soup.new_tag('th'))
    for ds in _DATASETS_ORDER:
        th = soup.new_tag('th', attrs={'colspan': '2'})
        th.string = {
            'RoadAnomaly21':         'SMIYC RA-21',
            'RoadObsticle21':        'SMIYC RO-21',
            'Fishyscapes Lost & Found': 'FS L&F',
            'Fishyscapes Static':    'FS Static',
            'RoadAnomaly':           'Road Anomaly',
        }[ds]
        tr1.append(th)
    thead.append(tr1)
    # Row 2: column sub-headers
    tr2 = soup.new_tag('tr')
    for label in ['Model', 'Method', 'mIoU']:
        th = soup.new_tag('th')
        th.string = label
        tr2.append(th)
    for _ in _DATASETS_ORDER:
        for sub in ['AuPRC', 'FPR95']:
            th = soup.new_tag('th')
            th.string = sub
            tr2.append(th)
    thead.append(tr2)
    table.append(thead)

    # --- tbody ---
    tbody = soup.new_tag('tbody')
    for model, methods in _TABLE_SCHEMA.items():
        for i, method in enumerate(methods):
            tr = soup.new_tag('tr')
            if i == 0:
                td_model = soup.new_tag('td', attrs={'rowspan': str(len(methods))})
                td_model.string = model
                tr.append(td_model)
            td_method = soup.new_tag('td')
            td_method.string = method
            tr.append(td_method)
            tr.append(_empty_td(soup))   # mIoU
            for _ in _DATASETS_ORDER:
                tr.append(_empty_td(soup))  # AuPRC
                tr.append(_empty_td(soup))  # FPR95
            tbody.append(tr)
    table.append(tbody)

    soup.append(table)
    return str(soup)


def _ensure_table():
    """Create or reset TABLE.md if missing / structurally invalid."""
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    needs_init = False

    if not os.path.exists(TABLE_PATH) or os.path.getsize(TABLE_PATH) == 0:
        needs_init = True
    else:
        with open(TABLE_PATH, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        if not soup.find('tbody'):
            needs_init = True

    if needs_init:
        html = _build_empty_table()
        with open(TABLE_PATH, 'w', encoding='utf-8') as f:
            f.write(html)


def update_table_entry(model: str, method: str, dataset: str = None, miou: str = None, auprc: str = None, fpr95: str = None):
    """
    Updates TABLE.md with the given metrics using BeautifulSoup.
    Auto-creates the file with empty placeholders if it does not exist or is malformed.
    - If only *miou* is provided, updates the mIoU column for the model/method row.
    - If *dataset*, *auprc*, and *fpr95* are provided, updates the respective AuPRC/FPR95 cells.
    """
    _ensure_table()

    with open(TABLE_PATH, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    tbody = soup.find('tbody')
    # _ensure_table guarantees tbody exists at this point
    if not tbody:
        raise ValueError("<tbody> not found in TABLE.md even after _ensure_table(); this is a bug.")

    current_model = None
    target_method = method.strip().lower()

    for tr in tbody.find_all('tr', recursive=False):
        tds = tr.find_all('td', recursive=False)
        if not tds:
            continue

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

                # Loose matching: "maxentropy" == "max entropy"
                is_match = (row_method == target_method) or \
                           (target_method == 'maxentropy' and row_method == 'max entropy')

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

    with open(TABLE_PATH, 'w', encoding='utf-8') as f:
        f.write(str(soup))
