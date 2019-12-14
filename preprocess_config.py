from pathlib import Path

dataset_root = 'data'
original_name = 'original' # path to Lakh dataset
tempo_name = 'tempo'
histo_name = 'histo'
song_histo_name = 'song_histo'
shifted_name = 'shifted'
roll_name = 'indroll'
chords_name = 'chords'
chords_index_name = 'chords_index'

data_dir = Path(dataset_root)
original_root_dir = data_dir / original_name
tempo_root_dir = data_dir / tempo_name
histo_root_dir = data_dir / histo_name
song_histo_root_dir = data_dir / song_histo_name
shifted_root_dir = data_dir / shifted_name
roll_root_dir = data_dir / roll_name
chords_root_dir = data_dir / chords_name
chords_index_root_dir = data_dir / chords_index_name


fs = 4
samples_per_bar = fs*2
octave = 12
melody_fs = 4


# Number of notes in extracted chords
num_notes_in_chord = 3
# Number of notes in a key
key_n = 7
# Chord Vocabulary size
num_chords = 100

double_sample_notes = False
UNK = '<unk>'