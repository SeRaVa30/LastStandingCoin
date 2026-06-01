import random

class action_space:
    def __init__(self, n):
        self.n = n  # Maksimal koin yang boleh diambil

class observation_space:
    def __init__(self, n):
        self.shape = (n,)  # Representasi dimensi sisa koin

class coin_game:
    """Environment untuk permainan Take Away Coin."""
    def __init__(self): 
        self.action_space = action_space(3)
        self.observation_space = observation_space(1)
        self.info = ""
        self.showboard = False
        self.reset()
        
    def reset(self):
        """Mengatur ulang permainan ke kondisi awal (30 koin)."""
        self.turn = 1       # Pemain 1 memulai duluan
        self.state = 30     # Jumlah koin awal
        self.validinputs = [1, 2, 3] if self.state > 3 else list(range(1, self.state + 1))
        self.done = False
        self.reward = 0
        self.move = 0
        return self.state
        
    def sample(self):
        """Mengambil aksi acak yang legal."""
        return random.choice(self.validinputs)
        
    def step(self, inp):
        """Mengeksekusi langkah, mereduksi state, dan mengevaluasi reward."""
        inp = int(inp)
        self.move = min(inp, self.state)
        self.state -= self.move  # Kurangi sisa koin di meja
        
        # Perbarui daftar langkah legal untuk giliran berikutnya
        self.validinputs = [1, 2, 3] if self.state > 3 else list(range(1, self.state + 1))
        
        # Evaluasi Kondisi Game Over & Pembagian Reward
        if self.state == 1:
            self.done = True
            self.reward = 1 if self.turn == 1 else -1
        elif self.state == 0:
            self.done = True
            self.reward = -1 if self.turn == 1 else 1
        else:
            self.turn = 2 if self.turn == 1 else 1  # Ganti giliran pemain
            
        return self.state, self.reward, self.done, self.info