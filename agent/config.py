class CFG():
    def __init__(self) -> None:
        self.N_EPISODES = 100

        self.CONV_SPIN_KERNEL = 64
        self.CONV_REP_KERNEL = 64
        
        self.SIZE_SPIN_KERNEL = 4
        self.SIZE_REP_KERNEL = 4
        
        self.DENSE_HIDDEN = 64
        self.DROPOUT_RATE = .2
        self.LSTM_SIZE = 64
        self.DEPTH = 1

        self.GAMMA = 0.2
        self.REPLAY_SIZE = 100000
        self.LEARNING_RATE = 2*1e-4

        self.ENTROPY_BETA= 0.25
        self.REWARD_STEPS = 1
        self.CLIP = 0.2

        self.N_GAMES = 5000*10
        self.DISPLAY_TIME = 1000
        self.EPOCHS = 100
        self.SYNC = 100
        
        self.rec = False