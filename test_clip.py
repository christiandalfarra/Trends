import os
import torch
import clip
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import json
import urllib.request

# CONFIGURAZIONE
DATASET_PATH = "../datasets/imagenet-adversarial/imagenet-a" 
MODEL_NAME = "ViT-B/16" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_imagenet_mapping():
    # Scarica la mappatura standard ImageNet (wnid -> nome classe)
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    try:
        with urllib.request.urlopen(url) as url:
            class_idx = json.loads(url.read().decode())
        # Crea dizionario: { 'n01440764': 'tench', ... }
        return {v[0]: v[1] for k, v in class_idx.items()}
    except Exception as e:
        print(f"Errore scaricamento mappatura: {e}")
        # Fallback: se fallisce il download, usa i nomi delle cartelle o un dizionario vuoto
        return {}

def main():
    print(f"Caricamento modello {MODEL_NAME} su {DEVICE}...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    
    # Mappatura classi (WNID -> Nome leggibile)
    wnid_to_name = load_imagenet_mapping()
    
    # Carica dataset CON la trasformazione (FONDAMENTALE PER EVITARE L'ERRORE)
    print(f"Caricamento immagini da {DATASET_PATH}...")
    dataset = ImageFolder(DATASET_PATH, transform=preprocess)
    
    # Prepara i prompt di testo
    # dataset.classes contiene i nomi delle cartelle (i codici WNID, es. n01440764)
    print("Generazione dei prompt di testo...")
    class_names = []
    for wnid in dataset.classes:
        # Se abbiamo il nome reale usiamo quello, altrimenti usiamo il codice WNID
        clean_name = wnid_to_name.get(wnid, wnid).replace('_', ' ')
        class_names.append(clean_name)

    # Tokenizza le classi
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(DEVICE)
    
    # Calcola embedding del testo (si fa una volta sola)
    print("Calcolo embedding del testo...")
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print("Inizio valutazione...")
    correct = 0
    total = 0
    
    # DataLoader
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

    accuracy = 100 * correct / total
    print(f"\nRisultati su {DATASET_PATH}:")
    print(f"Accuratezza Zero-Shot: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
