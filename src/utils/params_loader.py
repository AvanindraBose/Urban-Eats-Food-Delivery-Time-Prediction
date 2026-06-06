import yaml
import json
from pathlib import Path

def main():
    root_path = Path(__file__).parent.parent.parent
    xgb_params_path = root_path / "notebooks" / "xgb_best_params.json"
    rf_params_path = root_path / "notebooks" / "rf_best_params.json"
    params_path = root_path / "params.yaml"
    
    with open(xgb_params_path,"r") as f:
        xgb_best_params = json.load(f)
    
    with open(rf_params_path,"r") as f:
        rf_best_params = json.load(f)
    
    with open(params_path,"r") as f:
        params = yaml.safe_load(f)
    
    params["Train"] = {
    "XGBRegressor": xgb_best_params,
    "RFRegressor": rf_best_params
    }
    
    with open(params_path,"w") as f:
        yaml.dump(params,f)
    
    
if __name__ == "__main__":
    main()