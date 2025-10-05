# OCTMNIST Retinal Disease Classification

CNNs for **OCTMNIST** retinal disease classification. We implement two models:

- **BaseNN** – lightweight 2-conv CNN (Adam)
- **ImprovedNN** – deeper CNN with BatchNorm, Global Average Pooling, and Dropout (SGD + momentum + LR scheduler)

Both meet the course accuracy requirement; ImprovedNN trains more stably with slightly better test accuracy.  
(Background details and metrics summarized from my A0 report.) :contentReference[oaicite:0]{index=0}

---

## Dataset

We use **OCTMNIST** from **MedMNIST** (grayscale retinal OCT, 4 classes).  
The dataset auto-downloads on first run.

- Train: 97,477 images  
- Val:   10,832 images  
- Test:  1,000 images  
- Preprocessing: **resize to 64×64**, normalize to `[0,1]`, augment with random horizontal flip and small rotations (train only).  
- Note the **class imbalance** (class 3 most common; classes 1 & 2 underrepresented).

---

## Environments

```bash
python -m venv .env
source .env/bin/activate        # Windows: .env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
