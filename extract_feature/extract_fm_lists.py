import pandas as pd
from pathlib import Path

def generate_fm_lists():
    root = Path("/media/haoxiang/THX_HP_P900/Battery/dataset/Tao/Battery_Archive/outputs/soh_amotf_dino_runs/run_20260125_011032/bad_case_amotf_analysis")
    fm_csv = root / "bad_case_failure_modes.csv"
    
    if not fm_csv.exists():
        print("FM CSV not found")
        return

    df = pd.read_csv(fm_csv)
    
    # We want lists for FM3, FM4, FM5
    # failure_mode column might contain "FM3-FeatureCollapse/Smooth" etc.
    
    fm3 = df[df['failure_mode'].str.contains("FM3", na=False)]['sample_id'].unique()
    fm4 = df[df['failure_mode'].str.contains("FM4", na=False)]['sample_id'].unique()
    fm5 = df[df['failure_mode'].str.contains("FM5", na=False)]['sample_id'].unique()
    
    print(f"FM3 count: {len(fm3)}")
    print(f"FM4 count: {len(fm4)}")
    print(f"FM5 count: {len(fm5)}")
    
    # Output to a readable text file
    with open(root / "fm_sample_lists.txt", "w") as f:
        f.write(f"FM3 Samples ({len(fm3)}):\n")
        f.write(", ".join(fm3[:50])) # First 50
        f.write("\n...\n\n")
        
        f.write(f"FM4 Samples ({len(fm4)}):\n")
        f.write(", ".join(fm4[:50]))
        f.write("\n...\n\n")
        
        f.write(f"FM5 Samples ({len(fm5)}):\n")
        f.write(", ".join(fm5[:50]))
        f.write("\n...\n\n")
        
    # Also save full lists to csvs for the user
    pd.DataFrame(fm3, columns=["sample_id"]).to_csv(root / "fm3_samples.csv", index=False)
    pd.DataFrame(fm4, columns=["sample_id"]).to_csv(root / "fm4_samples.csv", index=False)
    pd.DataFrame(fm5, columns=["sample_id"]).to_csv(root / "fm5_samples.csv", index=False)

if __name__ == "__main__":
    generate_fm_lists()
