import numpy as np
import torch

def onehot_encoder(state):
    """Mengonversi integer state (sisa koin) menjadi vektor One-Hot Encoding."""
    onehot = np.zeros((1, 31), dtype=np.float32)  # Matriks nol 1x31 (rentang 0-30 koin)
    onehot[0, state] = 1                         # Set nilai 1 pada indeks sisa koin saat ini
    return onehot

def policy_agent(env, model, deterministic=True):
    """Inferensi model Deep Learning untuk memilih aksi (mode eksploitasi/eksplorasi)."""
    state = env.state
    
    # --- 1. Representasi Data ---
    onehot_state = onehot_encoder(state)
    state_tensor = torch.from_numpy(onehot_state)
    
    # --- 2. Inferensi Model (Forward Pass) ---
    model.eval()  # Set model ke mode evaluasi (matikan dropout/batchnorm)
    with torch.no_grad():
        logits = model(state_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()  # Distribusi probabilitas aksi
        
    # --- 3. Pemilihan Aksi ---
    if deterministic:
        action_idx = np.argmax(probs)  # Mode Eksploitasi: Pilih probabilitas tertinggi
        return action_idx + 1          # +1 untuk konversi indeks [0,1,2] menjadi koin [1,2,3]
    else:
        return np.random.choice([1, 2, 3], p=probs)  # Mode Eksplorasi: Acak berbasis peluang