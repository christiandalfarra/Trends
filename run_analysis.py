import os
import subprocess

# CONFIGURAZIONE
DATA_PATH = "../datasets" # <--- INSERISCI IL TUO PERCORSO QUI
DATASET = "A" # Usa "A" per ImageNet-A, "R" per ImageNet-R
ARCH = "ViT-B/16"

# Definisci gli esperimenti
# Chiave: nome esperimento, Valore: dizionario argomenti
experiments = {
    # 1. Variare le Viste (Augmentations)
    "views_16": {"--n_views": "16", "--steps": "1", "--selection_p": "0.1", "--lr": "5e-3"},
    "views_32": {"--n_views": "32", "--steps": "1", "--selection_p": "0.1", "--lr": "5e-3"},
    "views_64": {"--n_views": "64", "--steps": "1", "--selection_p": "0.1", "--lr": "5e-3"},
    
    # 2. Variare la Selection Confidence (Quanto filtrare le augmentation scarse)
    "conf_0.05": {"--n_views": "64", "--steps": "1", "--selection_p": "0.05", "--lr": "5e-3"},
    "conf_0.5":  {"--n_views": "64", "--steps": "1", "--selection_p": "0.5",  "--lr": "5e-3"},
    
    # 3. Variare Steps di ottimizzazione
    "steps_2":   {"--n_views": "64", "--steps": "2", "--selection_p": "0.1", "--lr": "5e-3"},
    
    # 4. Variare Learning Rate
    "lr_1e-2":   {"--n_views": "64", "--steps": "1", "--selection_p": "0.1", "--lr": "1e-2"},
}

def run_cmd(name, args):
    print(f"\n=== Running Experiment: {name} ===")
    cmd = [
        "python3", "tpt_classification.py",
        "--test_sets", DATASET,
        "-a", ARCH,
        "--data", DATA_PATH,
        "--gpu", "0"
    ]
    # Aggiungi gli argomenti specifici
    for k, v in args.items():
        cmd.append(k)
        cmd.append(v)
    
    # Salva output in un file log
    output_dir = f"../Trends/analysis_logs/{name}"
    os.makedirs(output_dir, exist_ok=True)
    cmd.append("--output")
    cmd.append(output_dir)
    
    print("Command:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    for exp_name, exp_args in experiments.items():
        run_cmd(exp_name, exp_args)
