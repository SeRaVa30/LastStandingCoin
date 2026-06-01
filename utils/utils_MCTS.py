from math import sqrt, log
from copy import deepcopy

def select(env, counts, wins, losses, temperature):
    """Fase Seleksi MCTS menggunakan formula Upper Confidence Bound (UCT)."""
    total_simulations = sum(counts.values())
    
    # Eksplorasi Prioritas: Coba langkah legal yang belum pernah dikunjungi
    for move in env.validinputs:
        if counts[move] == 0:
            return move
            
    scores = {}
    for move in env.validinputs:
        win_rate = (wins[move] - losses[move]) / counts[move]  # Eksploitasi
        exploration = temperature * sqrt(log(total_simulations) / counts[move])  # Eksplorasi
        scores[move] = win_rate + exploration
        
    return max(scores, key=scores.get)

def expand_and_simulate(env, move):
    """Fase Ekspansi & Simulasi Rollout dengan reward heuristik (4n+1)."""
    env_copy = deepcopy(env)  # Copy env agar tidak merusak state asli
    bonus_reward = 0
    
    state, reward, done, _ = env_copy.step(move)
    if state % 4 == 1 and not done:
        bonus_reward += 0.2  # Bonus jika berhasil mengunci posisi aman
    elif done:
        return reward
        
    while not done:
        random_move = env_copy.sample()
        state, reward, done, _ = env_copy.step(random_move)
        if state % 4 == 1 and not done:
            bonus_reward += 0.05
            
    return reward + bonus_reward

def backpropagate(move, reward, turn_at_start, counts, wins, losses):
    """Fase Backpropagation untuk memperbarui memori statistik pohon."""
    counts[move] += 1
    if (reward > 0 and turn_at_start == 1) or (reward < 0 and turn_at_start == 2):
        wins[move] += reward
    else:
        losses[move] += abs(reward)
    return counts, wins, losses

def mcts(env, num_rollouts, temperature):
    """Fungsi Utama Algoritma Monte Carlo Tree Search (MCTS)."""
    if len(env.validinputs) == 1:
        return env.validinputs[0]
        
    counts = {m: 0 for m in env.validinputs}
    wins = {m: 0 for m in env.validinputs}
    losses = {m: 0 for m in env.validinputs}
    current_turn = env.turn
    
    # Loop siklus penuh MCTS
    for _ in range(num_rollouts):
        move = select(env, counts, wins, losses, temperature)
        reward = expand_and_simulate(env, move)
        counts, wins, losses = backpropagate(move, reward, current_turn, counts, wins, losses)
        
    final_scores = {m: (wins[m] - losses[m]) / counts[m] if counts[m] > 0 else -float('inf') 
                    for m in env.validinputs}
    return max(final_scores, key=final_scores.get)