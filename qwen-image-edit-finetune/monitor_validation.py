#!/usr/bin/env python3
import time
from pathlib import Path

def monitor_validation():
    print("👀 MONITORING VALIDATION IN TRAINING")
    print("Watch for these validation indicators:")
    print("- '=== Running Validation at Step'")
    print("- 'Validating sample 1/4'")  
    print("- 'LPIPS ='")
    print("- 'Validation Summary'")
    
    log_file = Path("logs/segment_wandb_train_1398874.err")
    
    for i in range(60):  # Monitor for 5 minutes (check every 5 seconds)
        time.sleep(5)
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                content = f.read()
                
            if "=== Running Validation" in content:
                print("✅ VALIDATION DETECTED!")
                break
            elif "step=" in content:
                # Extract current step
                lines = content.split('\n')
                for line in lines[-5:]:
                    if 'step=' in line and 'Epoch' in line:
                        try:
                            step = line.split('step=')[1].split(',')[0].strip()
                            print(f"📊 Current step: {step}")
                        except:
                            pass

if __name__ == "__main__":
    monitor_validation()
