
import os, re, json
import numpy as np
import pandas as pd

# ---------------------------------------
# Meta utilities for AHI/TST (unchanged)
# ---------------------------------------
ID_RULES = {
    'shhs1': {'id_col': 'nsrrid', 'to_key': lambda x: str(int(x)).zfill(6), 'ahi': 'ahi_a0h3a', 'tst': 'slpprdp'},
    'shhs2': {'id_col': 'nsrrid', 'to_key': lambda x: str(int(x)).zfill(6), 'ahi': 'ahi_a0h3a', 'tst': 'slpprdp'},
    'mros1': {'id_col': 'nsrrid', 'to_key': lambda x: str(x).strip().lower(), 'ahi': 'nsrr_ahi_hp3r_aasm15', 'tst': 'nsrr_ttldursp_f1'},
    'mros2': {'id_col': 'nsrrid', 'to_key': lambda x: str(x).strip().lower(), 'ahi': 'poahi3a', 'tst': 'poslprdp'},
    'mesa' : {'id_col': 'mesaid', 'to_key': lambda x: str(int(x)).zfill(4), 'ahi': 'ahi_a0h3a', 'tst': 'slpprdp5'},
    'kiss' : {}  # JSON
}
FILE_MAP = {
    'shhs1': 'shhs1-dataset-0.20.0.csv',
    'shhs2': 'shhs2-dataset-0.20.0.csv',
    'mros1': 'mros-visit1-harmonized-0.6.0.csv',
    'mros2': 'mros-visit2-dataset-0.6.0.csv',
    'mesa' : 'mesa-sleep-dataset-0.7.0.csv',
}

def _safe_float(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan

def extract_key_from_any(pid, ds, rule):
    pid = str(pid)
    if ds == 'kiss': return pid
    if ds.startswith('mros'):
        m = re.search(r'(aa\d{4})', pid.lower())
        return m.group(1) if m else rule['to_key'](pid)
    if ds == 'mesa':
        m = re.search(r'(\d+)$', pid)
        return str(int(m.group(1))).zfill(4) if m else rule['to_key'](pid)
    if ds.startswith('shhs'):
        m = re.search(r'(\d{6})', pid)
        return m.group(1) if m else rule['to_key'](pid)
    return rule['to_key'](pid)

def extract_center_from_serial(serial_str):
    serial_str = serial_str.lower()
    if serial_str.startswith('mros-visit1'): return 'mros1'
    elif serial_str.startswith('mros-visit2'): return 'mros2'
    elif serial_str.startswith('shhs1'): return 'shhs1'
    elif serial_str.startswith('shhs2'): return 'shhs2'
    elif serial_str.startswith('mesa-sleep'): return 'mesa'
    elif re.match(r'^[a-dA-D]\d{4}-', serial_str): return 'kiss'
    else: raise ValueError(f"[extract_center_from_serial] Cannot parse center from serial: {serial_str}")

class AHI_TST_Loader:
    def __init__(self, meta_dir): self.meta_dir, self.cache = meta_dir, {}
    def load_nsrr_data(self, dataset):
        ds = dataset.lower()
        if ds.startswith('kiss'): return
        rule = ID_RULES[ds]
        path = os.path.join(self.meta_dir, FILE_MAP[ds])
        print(f"[{ds}] loading: {path}")
        df = pd.read_csv(path, encoding='cp1252', low_memory=False)
        for col in (rule['id_col'], rule['ahi'], rule['tst']):
            if col not in df.columns: raise KeyError(f"{col} not in {path}")
        df['__key__'] = df[rule['id_col']].apply(rule['to_key'])
        ahi = pd.to_numeric(df[rule['ahi']], errors='coerce')
        tst_min = pd.to_numeric(df[rule['tst']], errors='coerce')
        tst_h = tst_min.where(~(tst_min.isna() | (tst_min == 0)), np.nan) / 60.0
        for k, a, t in zip(df['__key__'], ahi, tst_h):
            self.cache[(ds, k)] = (a, t)
        print(f"  cached={len(df)}  NaN_AHI={ahi.isna().sum()}  NaN_TST={tst_h.isna().sum()}")

    def get(self, patient_id, dataset):
        ds = dataset.lower()
        if ds.startswith('kiss') or ds in ['a','b','c','d']:
            key = ('kiss', patient_id)
            if key in self.cache:
                return self.cache[key]
            path = os.path.join(self.meta_dir, "2020_annotation_full", f"{patient_id}_full_annotation.json")
            try:
                data = json.load(open(path))
                rep = data.get('Report', {})
                ahi = _safe_float(rep.get("AHI", np.nan))
                tst = _safe_float(rep.get("Total Sleep Time (TST)", np.nan))
                tst_h = np.nan if (tst in [None, 0] or np.isnan(tst)) else tst / 60
                self.cache[key] = (ahi, tst_h)
                return ahi, tst_h
            except FileNotFoundError:
                self.cache[key] = (np.nan, np.nan)
                return np.nan, np.nan
        rule = ID_RULES[ds]
        key = extract_key_from_any(patient_id, ds, rule)
        return self.cache.get((ds, key), (np.nan, np.nan))

    def to_dataframe(self, dataset=None):
        rows = []
        for (ds, key), (ahi, tst) in self.cache.items():
            if (dataset is None) or (ds == dataset.lower()):
                rows.append({'dataset': ds, 'key': key, 'AHI': ahi, 'TST_hours': tst})
        return pd.DataFrame(rows)

def load_all_nsrr_centers(ahi_loader, centers):
    for center in centers:
        try: ahi_loader.load_nsrr_data(center)
        except Exception as e: print(f"[ERROR] Failed to load {center}: {e}")

def collect_all_windows(dataset):
    if hasattr(dataset, 'windows'): return dataset.windows
    elif hasattr(dataset, 'datasets'):
        wins = []
        for ds in dataset.datasets:
            if hasattr(ds, 'windows'): wins.extend(ds.windows)
        return wins
    else:
        raise ValueError("Dataset has no .windows or .datasets attribute.")

def merge_serial_with_meta(data_windows, ahi_loader):
    ahi_dict, tst_dict, rows = {}, {}, []
    unique_pids = set()  # ← 윈도우 전체에서 환자 단 한 번씩만 처리

    for win in data_windows:
        serial = win.get('serial_numbers', [win.get('serial_number')])[0]
        pid = serial.split('/')[0]
        unique_pids.add(pid)

    for pid in unique_pids:
        try:
            center = extract_center_from_serial(pid)
            if center in {'kiss'}:           # ← KISS 계열은 JSON 경로
                ahi, tst = ahi_loader.get(pid, 'kiss')
                ahi_dict[pid], tst_dict[pid] = ahi, tst
            else:
                rule = ID_RULES[center]
                key = extract_key_from_any(pid, center, rule)
                rows.append({'Patient_ID': pid, 'center': center, 'key': key})
        except Exception as e:
            print(f"[WARN] Skipping serial parsing: {pid} ({e})")

    if rows:
        df = pd.DataFrame(rows)
        meta_all = ahi_loader.to_dataframe()
        merged = df.merge(meta_all, on=['key'], how='left').dropna(subset=['AHI', 'TST_hours'])
        ahi_dict.update(dict(zip(merged['Patient_ID'], merged['AHI'])))
        tst_dict.update(dict(zip(merged['Patient_ID'], merged['TST_hours'])))

    return ahi_dict, tst_dict