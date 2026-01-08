import shutil
from pathlib import Path

# Paths
WIDGET_CONTRACT_DIR = Path("/Users/jon/Desktop/sif-widget/shared/ai-form-contract")
LOCAL_CONTRACT_DIR = Path(__file__).resolve().parents[1] / "shared/ai-form-contract"

def import_contract():
    if not WIDGET_CONTRACT_DIR.exists():
        print(f"⚠️ Widget contract directory not found: {WIDGET_CONTRACT_DIR}")
        return

    print(f"🔄 Importing contract from {WIDGET_CONTRACT_DIR}...")
    
    if LOCAL_CONTRACT_DIR.exists():
        shutil.rmtree(LOCAL_CONTRACT_DIR)
    
    shutil.copytree(WIDGET_CONTRACT_DIR, LOCAL_CONTRACT_DIR)
    print(f"✅ Successfully imported contract to {LOCAL_CONTRACT_DIR}")

if __name__ == "__main__":
    import_contract()

