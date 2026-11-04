"""Unchanged Experiment 9 event vocabulary; inference-only constants."""
import hashlib

GENRES = ('classical','romantic','rock_pop')
GRID = 4
METERS = [(n,d) for d in (2,4,8,16) for n in range(1,13)]
TEMPOS = list(range(40,201,5))
TOKENS = (['PAD','BOS','EOS','BAR'] + [f'BPM_{n}' for n in TEMPOS]
    + [f'METER_{n}_{d}' for n,d in METERS]
    + [f'POS_{n}' for n in range(96)] + [f'PROGRAM_{n}' for n in range(129)]
    + [f'PITCH_{n}' for n in range(128)] + [f'DUR_{n}' for n in range(1,129)]
    + [f'VEL_{n}' for n in range(1,17)])
TO_ID = {t:i for i,t in enumerate(TOKENS)}
PAD, BOS, EOS, BAR = [TO_ID[t] for t in ('PAD','BOS','EOS','BAR')]
VOCAB_SHA = hashlib.sha256('\n'.join(TOKENS).encode()).hexdigest()
