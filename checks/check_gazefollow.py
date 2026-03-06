import os
from datasets import load_dataset
import matplotlib.pyplot as plt

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parquet_file = os.path.join(ROOT_DIR, "parquets", "gazefollow", "gazefollow_train.parquet")

print(f"📂 Chargement de {parquet_file}...")

# On charge le dataset comme le fera Swift
# Note: 'streaming=False' par défaut, donc il charge tout en RAM (ok pour un test)
try:
    ds = load_dataset("parquet", data_files={'train': parquet_file}, split='train')
    
    print(f"✅ Dataset chargé avec succès ! Nombre d'exemples : {len(ds)}")
    
    # --- TEST DE LECTURE DU PREMIER EXEMPLE ---
    index_to_test = 0  # Tu peux changer pour voir d'autres exemples
    example = ds[index_to_test]

    print("\n--- 📝 CONTENU TEXTUEL (Messages) ---")
    for msg in example['messages']:
        print(f"[{msg['role']}] : {msg['content'][:100]}...") # On coupe si c'est trop long

    print("\n--- 🖼️ CONTENU IMAGE ---")
    images = example['images']
    
    if images and len(images) > 0:
        img = images[0]
        print(f"Type de l'objet : {type(img)}") 
        print(f"Taille : {img.size}")
        print("Affichage de l'image en cours...")
        
        # Ouvre l'image dans ta visionneuse par défaut
        img.show() 
        
        # Si img.show() ne marche pas, tu peux décommenter les lignes ci-dessous :
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
    else:
        print("⚠️ Aucune image trouvée dans cet exemple.")

except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")