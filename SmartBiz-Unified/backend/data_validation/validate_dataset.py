import os
import pandas as pd
import json
import gc

class DatasetValidator:
    def __init__(self, data_path: str, report_dir: str):
        self.data_path = data_path
        self.report_dir = report_dir
        self.files = {
            'sales': 'sales_train_validation.csv',
            'calendar': 'calendar.csv',
            'prices': 'sell_prices.csv'
        }
        self.report = {}
        
    def validate(self) -> bool:
        print("Starting Dataset Validation...")
        all_passed = True
        
        for key, filename in self.files.items():
            file_path = os.path.join(self.data_path, filename)
            file_passed, file_report = self._validate_file(file_path, key)
            self.report[filename] = file_report
            if not file_passed:
                all_passed = False
                
        self._save_report()
        
        if not all_passed:
            print("[ERROR] Dataset validation FAILED. See report for details.")
            raise ValueError("Dataset validation failed due to missing files or data inconsistencies. Please check the logs and report.")
        else:
            print("[SUCCESS] Dataset validation PASSED.")
            
        return all_passed

    def _validate_file(self, file_path: str, file_type: str):
        report = {
            'exists': False,
            'shape': None,
            'missing_values': None,
            'duplicated_rows': None,
            'schema': {}
        }
        
        if not os.path.exists(file_path):
            print(f"[ERROR] Missing file: {file_path}")
            return False, report
            
        report['exists'] = True
        print(f"Validating {file_path}...")
        
        try:
            # We load the CSV to check shape, missing values, duplicates, and schema.
            # For sales, we might encounter high RAM usage, so we'll collect garbage aggressively.
            df = pd.read_csv(file_path)
            
            report['shape'] = list(df.shape)
            report['missing_values'] = int(df.isna().sum().sum())
            report['duplicated_rows'] = int(df.duplicated().sum())
            
            for col in df.columns:
                report['schema'][col] = str(df[col].dtype)
                
            del df
            gc.collect()
            
            # Simple integrity rules
            if report['missing_values'] > 0 and file_type == 'sales':
                print(f"[WARNING] Found {report['missing_values']} missing values in {file_type}.")
                
            return True, report
            
        except Exception as e:
            print(f"[ERROR] Failed to validate {file_path}: {e}")
            report['error'] = str(e)
            return False, report
            
    def _save_report(self):
        os.makedirs(self.report_dir, exist_ok=True)
        report_path = os.path.join(self.report_dir, 'dataset_validation_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=4)
        print(f"Validation report saved to {report_path}")
        
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    probe_file = 'sales_train_validation.csv'
    top_candidates = [
        os.path.abspath(os.path.join(base_dir, '..', 'SmartBiz-Dataset v1.0')),
        os.path.abspath(os.path.join(base_dir, '..', 'SmartBiz-Dataset-v1.0')),
    ]
    data_dir = None
    for top in top_candidates:
        if not os.path.isdir(top):
            continue
        if os.path.exists(os.path.join(top, probe_file)):
            data_dir = top
            break
        for entry in os.scandir(top):
            if entry.is_dir() and os.path.exists(os.path.join(entry.path, probe_file)):
                data_dir = entry.path
                break
        if data_dir:
            break

    if not data_dir:
        print(f"[ERROR] Could not locate dataset. Checked: {top_candidates}")
        exit(1)

    report_dir = os.path.join(base_dir, 'experiments', 'evaluation_results')
    validator = DatasetValidator(data_dir, report_dir)
    validator.validate()
