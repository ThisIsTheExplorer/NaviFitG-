# suara.py
import os, pygame, time

pygame.mixer.pre_init(44100, -16, 2, 512)
pygame.init()

def find_audio(base):
    for ext in ("wav","ogg","mp3"):
        p = f"{base}.{ext}"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"File {base}.wav/.ogg/.mp3 tidak ditemukan")

fn = find_audio("kiri")       # akan cari kiri.wav lalu kiri.ogg, kiri.mp3
s  = pygame.mixer.Sound(fn)   # untuk klip pendek
ch = s.play()
while ch.get_busy():
    time.sleep(0.05)

pygame.quit()
