def rule_based_AI(env):
    """Agen Heuristik berbasis Teori Permainan Nim untuk mencari posisi aman."""
    if env.state % 4 != 1:
        move = (env.state - 1) % 4  # Paksa sisa koin di meja menjadi kelipatan (4n + 1)
    else:
        move = env.sample()         # Langkah acak jika agen terjebak di posisi kalah
    return move

def random_player(env):
    """Agen Baseline (Kontrol) yang mengambil langkah sepenuhnya secara acak."""
    move = env.sample()             # Ambil aksi legal apa saja tanpa strategi
    return move