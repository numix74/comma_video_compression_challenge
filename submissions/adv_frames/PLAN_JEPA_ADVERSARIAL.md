# Plan : JEPA-Adversarial Compression

## Concept

Optimiser les frames dans l'espace latent V-JEPA (lisse, sémantique) plutôt
que dans l'espace pixel. La convergence est ~10x plus rapide. Un tiny decoder
(dans l'archive) mappe les latents optimisés → RGB.

## Différence vs Adversarial Frames

| | Adversarial Frames | JEPA-Adversarial |
|---|---|---|
| Espace d'optimisation | Pixel (H×W×3 = 3M dims) | Latent JEPA (196×1280 = 250K dims) |
| Lisseur naturel | Pénalité HF explicite | Encodeur JEPA filtre nativement le HF |
| Codec loop | Nécessaire (AV1 casse pixels) | Moins critique (latents sont stables) |
| Dans l'archive | Vidéo + poses | Latents compressés + tiny decoder + poses |

## Architecture

### Compression (offline, pas dans l'archive)

```
V-JEPA Encoder (ViT-H, 600M params, HuggingFace)
  ↓
Latents : (N, 196, 1280) par frame
  ↓
VQ (Vector Quantization) : codebook 4096 entrées × 256D
  ↓
Tokens discrets : (N, 196) entiers ∈ [0, 4095]
  ↓
Compression : AV1 sur la "vidéo de tokens" (grayscale 16-bit) + Brotli
```

### Archive (ce qui est stocké)
- `tokens.br` : tokens VQ compressés (~20-40KB estimé)
- `poses.npy.br` : poses GT (~5KB)
- `decoder.pt.br` : tiny decoder FP4 (~15-20KB)

**Total estimé : ~45-65KB**  
**Rate : 45/37545 = 0.0012 → 25×0.0012 = 0.030**

### Tiny Decoder (dans l'archive)

```
Input  : token indices (196 tokens) → embedding lookup (4096×256)
         + pose (6D) → Linear(6, 64)
Body   : SepConv UNet (3 niveaux, ~200K params)
         Upsampling bilinéaire
Output : RGB frame (874×1164×3)
Loss   : CE(SegNet) + MSE(PoseNet) + λ·L2_pixel
```

Architecture légère : ~200K params → ~100KB fp32 → ~25KB FP4+Brotli

### Inflation (aucun GPU requis, < 2 minutes)

```python
tokens = brotli_decompress("tokens.br")  # (N, 196)
poses  = brotli_decompress("poses.npy.br")
decoder = load_fp4("decoder.pt.br")      # tiny model

for i in range(N):
    emb = codebook[tokens[i]]   # (196, 256)
    frame = decoder(emb, poses[i//2])  # (874, 1164, 3)
    write_raw(frame)
```

## Pipeline d'entraînement

### Étape 1 : Setup (1 jour)
```bash
# Sur Vast.ai A100
pip install git+https://github.com/facebookresearch/jepa
# Télécharger V-JEPA ViT-H weights (~2.4GB)
# Télécharger test_videos.zip (2.4GB, 64 vidéos)
```

### Étape 2 : Encodage JEPA de toutes les vidéos (2-4h GPU)
```python
# Encoder les 64 vidéos × ~1200 frames en latents V-JEPA
# Construire le codebook VQ (k-means sur tous les latents)
# Encoder les tokens → compression AV1 + Brotli
```

### Étape 3 : Entraînement du tiny decoder (6-12h GPU)
```python
# Dataset : (tokens, pose_gt) → frame_rgb
# Loss : CE(SegNet(reconstruit)) + MSE(PoseNet(reconstruit)) + L2_pixel
# QAT : FP4 dès epoch 50
# Eval : score challenge toutes les 10 epochs
```

### Étape 4 : Optimisation adversariale dans l'espace latent (1-2h GPU)
```python
# Pour chaque frame de 0.mkv :
# Fine-tuner les tokens (gradient descent dans l'espace VQ continu)
# pour minimiser CE(SegNet) + MSE(PoseNet) directement
# Puis re-quantizer → tokens finaux stockés
```

## Estimation de score

| Composante | Valeur | Points |
|---|---|---|
| Rate (50KB/37.5MB) | 0.00133 | 0.033 |
| SegNet (decoder bien entraîné) | ~0.001-0.003 | 0.1-0.3 |
| PoseNet (poses stockées + decoder) | ~0.001-0.005 | 0.1-0.22 |
| **Total** | | **~0.23-0.55** |

Optimiste : **0.15** / Réaliste : **0.30** / Pessimiste : **0.60**

## Risques principaux

1. **Latents V-JEPA → pixels** : V-JEPA n'est pas entraîné pour la reconstruction
   pixel, ses latents sont abstraits. Le decoder doit apprendre ce mapping.
   → Mitigation : perte pixel (L2) en plus des pertes sémantiques

2. **Codebook VQ instable** : si le VQ collapse (tous les tokens vers 1 entrée)
   → Mitigation : commitment loss + entropy regularization

3. **Decoder trop petit** : 200K params peut être insuffisant
   → Si nécessaire, augmenter à 500K (~50KB FP4) → rate = 0.035

## Décision de bifurcation

Passer à JEPA-Adversarial si :
- Adversarial Frames score > 0.25 après optimisation complète
- OU si la boucle codec AV1 est trop instable en pratique

## Prérequis matériel

- Compress : A100 80GB idéal (ou 2× A100 40GB), ~12-18h total
- Inflate : CPU pur, < 2 minutes
