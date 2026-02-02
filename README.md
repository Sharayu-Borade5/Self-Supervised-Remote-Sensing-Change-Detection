# Self-Supervised Remote Sensing Change Detection (SS-RSCD)

Detect changes in satellite imagery without labeled change masks.

## 🌍 Motivation
Labeled change detection datasets are expensive and scarce.
SS-RSCD learns temporal representations using contrastive learning
and detects changes via feature divergence.

## ✨ Key Features
- No pixel-level labels
- Siamese CNN encoder
- Contrastive temporal learning
- Unsupervised change maps
- Urban growth & disaster analysis

## 🧠 Method
Two temporally separated images are encoded into a shared latent space.
Significant embedding divergence indicates change.

## 📊 Applications
- Urban expansion
- Flood & wildfire damage
- Infrastructure monitoring

## 🚀 Quick Start
```bash
pip install -r requirements.txt
python demo.py
