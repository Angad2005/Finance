import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_wfv_splits(total_days, train_window=752, val_window=63, test_window=63, output_dir=None):
    splits = []
    current_end = train_window
    
    while current_end + val_window + test_window <= total_days:
        train_start = 0  # Expanding window
        train_end = current_end
        
        val_start = train_end
        val_end = val_start + val_window
        
        test_start = val_end
        test_end = test_start + test_window
        
        # Validation checks
        assert max(range(train_start, train_end)) < min(range(val_start, val_end)), "Train and Val overlap!"
        assert max(range(val_start, val_end)) < min(range(test_start, test_end)), "Val and Test overlap!"
        
        splits.append({
            "train": (train_start, train_end),
            "val": (val_start, val_end),
            "test": (test_start, test_end)
        })
        
        current_end += test_window
        
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        splits_path = output_dir / "wfv_splits.json"
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=4)
        logger.info(f"Saved {len(splits)} WFV splits to {splits_path}")
        
    return splits
