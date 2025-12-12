import os
import torch
import clip
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import json
import urllib.request

# CONFIGURAZIONE
DATASET_PATH = "../datasets
/imagenet-adversarial/imagenet-a" # Cambia con il tuo percorso
MODEL_NAME = "ViT-B/32" # o "RN50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_imagenet_mapping():
    # Scarica la mappatura standard ImageNet (wnid -> nome classe)
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    with urllib.request.urlopen(url) as url:
        class_idx = json.loads(url.read().decode())
    # Crea dizionario: { 'n01440764': 'tench', ... }
    return {v[0]: v[1] for k, v in class_idx.items()}

def main():
    print(f"Caricamento modello {MODEL_NAME} su {DEVICE}...")
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    
    # Mappatura classi
    wnid_to_name = load_imagenet_mapping()
    
    # Carica dataset
    print(f"Caricamento immagini da {DATASET_PATH}...")
    dataset = ImageFolder(DATASET_PATH) # ImageFolder usa i nomi delle cartelle (wnid) come classi
    
    # Prepara i prompt di testo (solo per le classi presenti nel dataset A o R)
    # dataset.classes contiene i wnid (es. n01440764) presenti nella cartella
    class_names = [wnid_to_name[wnid].replace('_', ' ') for wnid in dataset.classes]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).to(DEVICE)
    
    print("Calcolo embedding del testo...")
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print("Inizio valutazione...")
    correct = 0
    total = 0
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE) # Questi sono indici da 0 a 199 (per ImageNet-A/R)
            
            # Preprocess e Encode immagini
            # Nota: le immagini sono già trasformate dal loader se usiamo il transform di CLIP? 
            # ImageFolder di default non ha transform, dobbiamo passarglielo.
            # Fix manuale: applichiamo preprocess singolarmente o usiamo un wrapper.
            # Per semplicità qui sopra ho dimenticato il transform nel dataset. 
            pass 

    # CORREZIONE: ImageFolder deve avere il preprocess
    dataset = ImageFolder(DATASET_PATH, transform=preprocess)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Calcola similarità
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predictions = similarity.argmax(dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"\nAccuratezza Zero-Shot: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
