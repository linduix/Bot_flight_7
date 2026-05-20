from Trainer import show_archive, save_path
from modules.evo_alg.mapElites import load
import os

if os.path.isfile(save_path):
    alg, _, _ = load(save_path)
    show_archive(alg)
else:
    print('Archive Not Found')