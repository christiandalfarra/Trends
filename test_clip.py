import os
import torch
import clip
import argparse
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import json
import urllib.request
import sys

# CONFIGURAZIONE MODELLO
MODEL_NAME = "ViT-B/16" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = argparse.ArgumentParser(description="Test Zero-Shot CLIP su un dataset locale.")
    # L'argomento 'dataset_path' è obbligatorio e posizionale
    parser.add_argument("dataset_path", type=str, help="Il percorso alla root del dataset (es. path/to/imagenet-a)")
    return parser.parse_args()

def load_imagenet_mapping():
    """Scarica la mappatura standard ImageNet (wnid -> nome classe)"""
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    try:
        with urllib.request.urlopen(url) as url:
            class_idx = json.loads(url.read().decode())
        # Crea dizionario: { 'n01440764': 'tench', ... }
        return {v[0]: v[1] for k, v in class_idx.items()}
    except Exception as e:
        print(f"Attenzione: Impossibile scaricare la mappatura ImageNet ({e}).")
        print("Verranno usati i nomi delle cartelle (WNID) come etichette.")
        return {}

def main():
    # 1. Parsing degli argomenti da terminale
    args = get_args()
    dataset_path = args.dataset_path

    # Controllo di sicurezza sul percorso
    if not os.path.exists(dataset_path):
        print(f"ERRORE: La cartella '{dataset_path}' non esiste.")
        sys.exit(1)

    print(f"Configurazione: Modello {MODEL_NAME} su {DEVICE}")
    print(f"Dataset target: {dataset_path}")

    # 2. Caricamento Modello CLIP
    print("Caricamento modello...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    
    # 3. Mappatura classi (WNID -> Nome leggibile)
    wnid_to_name = load_imagenet_mapping()
    
    # 4. Carica dataset
    # IMPORTANTE: transform=preprocess gestisce la conversione da PIL a Tensor
    try:
        dataset = ImageFolder(dataset_path, transform=preprocess)
    except FileNotFoundError:
        print("ERRORE: Struttura cartelle non valida. Assicurati che il percorso contenga sottocartelle per le classi.")
        sys.exit(1)
    
    # 5. Prepara i prompt di testo
    print("Generazione dei prompt di testo...")
    class_names = []
    # dataset.classes contiene i nomi delle sottocartelle (es. n01440764)
    for wnid in dataset.classes:
        # Se il wnid è nel dizionario, prendi il nome, sostituisci _ con spazi. Altrimenti tieni il wnid.
        clean_name = wnid_to_name.get(wnid, wnid).replace('_', ' ')
        class_names.append(clean_name)

    # Tokenizza le classi
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(DEVICE)
    
    # 6. Calcola embedding del testo
    print("Calcolo embedding del testo...")
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 7. Ciclo di Valutazione
    print("Inizio valutazione...")
    correct = 0
    total = 0
    
    # DataLoader ottimizzato
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Encode immagini
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calcola similarità (prodotto scalare)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Prendi l'indice con la probabilità più alta
            predictions = similarity.argmax(dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # 8. Risultati
    if total > 0:
        accuracy = 100 * correct / total
        print(f"\n=== RISULTATI ===")
        print(f"Dataset: {dataset_path}")
        print(f"Immagini totali: {total}")
        print(f"Accuratezza Zero-Shot: {accuracy:.2f}%")
    else:
        print("Nessuna immagine trovata o elaborata.")

if __name__ == "__main__":
    main()
