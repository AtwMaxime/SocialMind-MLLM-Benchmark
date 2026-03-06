import os
import json
from datasets import Dataset, Features, Sequence, Value, Image

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = {
    "train":      os.path.join(ROOT_DIR, "dataset", "gazefollow", "gazefollow_train_ms_swift.jsonl"),
    "validation": os.path.join(ROOT_DIR, "dataset", "gazefollow", "gazefollow_val_ms_swift.jsonl"),
}

# Root folder where GazeFollow images are stored on your machine — edit this to match your setup
LOCAL_ROOT = "/home/mattwood/Bureau/SocialFlorence/"

# Jean Zay path prefix to strip when remapping image paths
JEAN_ZAY_PREFIX = "/lustre/fswork/projects/rech/tey/uvu79wi/gazeVLM/"

OUTPUT_DIR = os.path.join(ROOT_DIR, "parquets", "gazefollow")

# ==========================================
# 2. FONCTIONS UTILES
# ==========================================

def get_local_path(jz_path):
    """Convertit le chemin Jean Zay en chemin local."""
    if not jz_path.startswith(JEAN_ZAY_PREFIX):
        # Si le path ne commence pas par le préfixe attendu, on le garde tel quel
        # (au cas où tu aurais déjà modifié le jsonl)
        return jz_path
    
    # On enlève le préfixe Jean Zay et on colle le préfixe Local
    relative_path = jz_path.replace(JEAN_ZAY_PREFIX, "")
    local_path = os.path.join(LOCAL_ROOT, relative_path)
    return local_path

def generator(jsonl_path):
    """Lit le JSONL et génère les données avec images binaires."""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                
                # --- Gestion des Images ---
                images_out = []
                original_paths = data.get("images", [])
                if isinstance(original_paths, str): original_paths = [original_paths]
                
                valid_entry = True
                
                for path in original_paths:
                    local_p = get_local_path(path)
                    
                    if os.path.exists(local_p):
                        # On lit l'image en binaire et on la stocke
                        # HF datasets gérera le décodage grâce à Image(decode=True)
                        with open(local_p, "rb") as img_f:
                            images_out.append({"bytes": img_f.read(), "path": None})
                    else:
                        print(f"⚠️ Image introuvable (Ligne {i}) : {local_p}")
                        print(f"   (Original: {path})")
                        valid_entry = False
                        break # Si une image manque, on jette l'exemple (optionnel)
                
                # Si toutes les images sont là, on garde l'exemple
                if valid_entry:
                    yield {
                        "messages": data["messages"],
                        "images": images_out
                    }
                    
            except Exception as e:
                print(f"❌ Erreur ligne {i}: {e}")

# ==========================================
# 3. CRÉATION DU DATASET
# ==========================================

# Définition du schéma
# Image(decode=True) est crucial : il stocke les bytes et renverra un objet PIL.Image au chargement
features = Features({
    "messages": [{"role": Value("string"), "content": Value("string")}],
    "images": Sequence(Image(decode=True))
})

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split_name, file_path in FILES.items():
    print(f"\n🚀 Traitement de {split_name} depuis {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"❌ Fichier JSONL introuvable : {file_path}")
        continue

    # Création du dataset
    ds = Dataset.from_generator(
        generator, 
        gen_kwargs={"jsonl_path": file_path}, 
        features=features
    )
    
    # Sauvegarde
    save_path = os.path.join(OUTPUT_DIR, f"gazefollow_{split_name}.parquet")
    ds.to_parquet(save_path)
    
    print(f"✅ Sauvegardé : {save_path}")
    print(f"   (Taille : {len(ds)} exemples)")

print("\n✨ Terminé pour GazeFollow !")