from typing import Union, List, Dict
from src.utils.conf_instance import Instance
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# == Training == ##########################
seed: int = 96
deterministic: bool = False

# CPU, GPU, TPU
gpus: Union[int, None] = 1
tpu_cores: Union[int, None] = None

# Epoch or Step
max_epochs: int = 40
val_check_interval: int = 1  # Validate after each epoch
# Learning rate scheduler
#num_warmup_steps: int = 1  # Warmup steps
#max_steps: int = 5       # Total training steps

# Batch
batch_size: int = 16  

# Gradient
accumulate_grad_batches: int = 2 
gradient_clip_val: float = 1.0
weight_decay: float = 0.01
lr: float = 1e-5
optimizer: str = "adamw"

# Precision
precision: int = 32  # Mixed precision

# Pretrained model
pmodel: Dict = {
    "name": "microsoft/deberta-large",
    "max_len": 256,  
}

example_sentences = {
    0: [  # Literal
        "He walked to the store.",
        "She is reading a book.",
        "They are playing outside.",
        "The teacher explained the lesson.",
        "She cooked dinner for her family.",
        "He parked the car in the garage.",
        "The cat is sleeping on the couch.",
        "They enjoyed a quiet evening at home.",
        "The boy completed his homework.",
        "She took the bus to work."
    ],

    1: [  # Metaphor
        "The economy is on the brink of collapse.",
        "His career is a sinking ship.",
        "The stock market is a roller coaster ride.",
        "Their friendship burned out like a candle in the wind.",
        "The city is a jungle at night.",
        "His words were a dagger to my heart.",
        "The waves of change swept through the nation.",
        "Time is a thief that steals our youth.",
        "But Mr. Quayle can't seem to escape controversy.",
        "Friends say a 1975 auto crash... seared new priorities into his management approach.",
        "The marauding huns of the takeover game are besieging ever-larger corporate kingdoms.",
        "Money that escapes the pockets of bureaucrats disappears into rat holes.",
        "The World Bank effectively is stepping into the breach left by a recalcitrant IMF.",

    ],

    2: [  # Metonymy
        "The White House issued a statement today.",
        "Wall Street is nervous about inflation.",
        "Hollywood loves a scandal.",
        "Silicon Valley is innovating rapidly.",
        "The Pentagon announced new defense strategies.",
        "The Vatican made an official declaration.",
        "Downing Street responded to the allegations.",
        "Broadway is full of excitement this season.",
        "The Kremlin denied the accusations.",
        "Capitol Hill is in turmoil over the new bill."
    ],

    3: [  # Simile
        "She runs like the wind.",
        "His smile is as bright as the sun.",
        "The baby was like an angel sleeping.",
        "Her voice is like honey dripping from a spoon.",
        "The clouds are like cotton candy in the sky.",
        "He fought like a lion in the battle.",
        "The car moved as fast as a cheetah.",
        "Her eyes sparkled like diamonds.",
        "His temper is like a thunderstorm.",
        "The mountain stood tall like a guardian."
    ]
}



# Dataset Paths
train_path: str = "/home/.../FLUTE/cleandata/train3.6.txt"
val_path: str = "/home/.../FLUTE/cleandata/val3.txt"
test_path: str = "/home/.../FLUTE/cleandata/test3.txt"
test_datasets = {
    "ovrall":"/home/.../FLUTE/cleandata/test.txt",
    "MOH-X": "/home/.../FLUTE/cleandata/MOH-X-test.txt",
    "TroFi": "/home/.../FLUTE/cleandata/TroFi-test.txt",
    "VUA20":"/home/.../FLUTE/cleandata/VUA20_filtered_test2.txt",
    "VUA18":"/home/.../FLUTE/cleandata/VUA18_filtered_test.txt",
    "VUAverb":"/home/.../FLUTE/cleandata/VUAverb-test.txt",
    "relocalr": "/home/.../FLUTE/cleandata/relocalr-test.txt",
    "simile": "/home/.../FLUTE/cleandata/smile-test.txt",
}

# Callbacks
callbacks_monitor: str = "val_loss"
callbacks_mode: str = "min"
