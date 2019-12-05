import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import mido
import numpy as np
import pretty_midi as pm
from tqdm import tqdm

import config


def is_containing_data_directly(data_dir):
    """Return boolian of which data is directly under data_dir for Lakh data dir format."""
    return data_dir.is_dir() and len(data_dir.name) != 1


def process_dir(original_dir, target_dir, original_suffix, target_suffix, process_file, **kwargs):
    for original_file in original_dir.glob('*.' + original_suffix):
        try:
            process_file(original_file,
                         target_dir / original_file.name.replace(original_suffix, target_suffix), **kwargs)
        except (ValueError, OSError) as e:
            exception_str = 'Unexpected error in ' + str(original_file)  + ':\n', e, sys.exc_info()[0]
            print(exception_str)


def generate_target_from_original(original_root_dir, target_root_dir,
                                  original_suffix, target_suffix, process_file, **kwargs):
    """Util func for processing original data to target data.
        Args:
            original_root_dir (pathlib object): Path to root dir of Lakh data dir format, like `lmd_matched`.
            target_root_dir (pathlib objext): Path to root dir of Lakh data dir format. Usually it is empty before working this func.
            original_suffix (str): Data suffix before processed.
            target_suffix (str): Data suffix after processed.
            process_file (func): Func for processing each data(.original_suffix).
            (**kwargs: Optional args for process_file())
    """
    for original_dir in tqdm(original_root_dir.glob('**/*')):
        if is_containing_data_directly(original_dir):
            target_dir = Path(str(original_dir).replace(original_root_dir.name, target_root_dir.name))
            if not(target_dir.exists()):
                target_dir.mkdir(parents=True)
            process_dir(original_dir, target_dir, original_suffix, target_suffix, process_file, **kwargs)


def change_tempo_midi(midi_file, save_path):
    mid = mido.MidiFile(midi_file)
    new_mid = mido.MidiFile()

    new_mid.ticks_per_beat = mid.ticks_per_beat
    for track in mid.tracks:
        new_track = mido.MidiTrack()
        for msg in track:
            new_msg = msg.copy()
            if new_msg.type == 'set_tempo':
                new_msg.tempo = 500000
            new_track.append(new_msg)
        new_mid.tracks.append(new_track)
    new_mid.save(save_path)



def pianoroll_to_histo_bar(pianoroll, samples_per_bar):
    bar_num = pianoroll.shape[1] // samples_per_bar
    histo_bar = np.zeros((pianoroll.shape[0], bar_num))
    for i in range(bar_num):
        histo_bar[:, i] = np.sum(pianoroll[:, i*samples_per_bar:(i+1)*samples_per_bar], axis=1)
    return histo_bar


def histo_bar_to_histo(histo_bar, octave):
    histo_oct = np.zeros((octave, histo_bar.shape[1]))
    octave_num = histo_bar.shape[0] // octave
    for i in range(octave_num - 1):
        histo_oct = np.add(histo_oct, histo_bar[i*octave:(i+1)*octave])
    return histo_oct


def midi_to_histo(midi_file, save_path):
    """hist_oct[i][j]: non zero num per bar_j for key_i"""
    mid = pm.PrettyMIDI(str(midi_file))
    pianoroll = mid.get_piano_roll()  # rethink to use original get_piano_roll func
    histo_bar = pianoroll_to_histo_bar(pianoroll, samples_per_bar)
    histo_oct = histo_bar_to_histo(histo_bar, octave)  # shape: (12, pianoroll.shape[1]_per_song // samples_par_bar)
    pickle.dump(histo_oct, open(str(save_path), 'wb'))
        
        
generate_target_from_original(original_root_dir=tempo_root_dir,
                              target_root_dir=histo_root_dir,
                              original_name=tempo_name,
                              target_name=histo_name,
                              original_suffix='mid',
                              target_suffix='pickle',
                              process_file=midi_to_histo)

def histo_to_song_histo(histo_file, save_path):
    histo = pickle.load(open(str(histo_file), 'rb'))
    song_histo = np.sum(histo, axis=1)
    pickle.dump(song_histo, open(save_path , 'wb'))

generate_target_from_original(original_root_dir=histo_root_dir,
                              target_root_dir=song_histo_root_dir,
                              original_name=histo_name,
                              target_name=song_histo_name,
                              original_suffix='pickle',
                              target_suffix='pickle',
                              process_file=histo_to_song_histo)

def pianoroll_to_note_index(pianoroll):
    note_ind = []
    for i in range(0, pianoroll.shape[1]):
        step = []
        for j, note in enumerate(pianoroll[:,i]):
            if note != 0:
                step.append(j)
        note_ind.append(tuple(step))
    return note_ind


def note_to_index(midi_file, save_path):
    mid = pm.PrettyMIDI(str(midi_file))
    p = mid.get_piano_roll(fs=fs)

    # make note value 1, Todo
    for i, _ in enumerate(p):
        for j, _ in enumerate(p[i]):
            if p[i, j] != 0:
                p[i,j] = 1

    n = pianoroll_to_note_index(p)
    pickle.dump(n, open(str(save_path), 'wb'))

    
generate_target_from_original(original_root_dir=tempo_root_dir,
                              target_root_dir=roll_root_dir,
                              original_name=tempo_name,
                              target_name=roll_name,
                              original_suffix='mid',
                              target_suffix='pickle',
                              process_file=note_to_index)

def histo_to_chord(histo_file, save_path):
    histo = pickle.load(open(histo_file, 'rb'))
    max_n = histo.argsort(axis=0)[-chord_n:]
    chords = []
    for i in range(0, max_n.shape[1]):
        chord = []
        for note in max_n[:,i]:
            if histo[note, i] != 0:  # ここがなんでいるかわからない
                chord.append(note)
            # chord.append(note)
        chord.sort()
        chords.append(tuple(chord))
    assert(histo.shape[1] == len(chords))
    # print(chords)
        
    pickle.dump(chords, open(str(save_path), 'wb'))
        
generate_target_from_original(original_root_dir=histo_root_dir,
                              target_root_dir=chords_root_dir,
                              original_name=histo_name,
                              target_name=chords_name,
                              original_suffix='pickle',
                              target_suffix='pickle',
                              process_file=histo_to_chord)


def get_chord_dict():
    chord_to_index = pickle.load(open("data/chord_to_index.pickle", 'rb'))
    index_to_chord = pickle.load(open("data/index_to_chord.pickle", 'rb'))
    return chord_to_index, index_to_chord


def chords_to_index(chords,chord_to_index):
    chords_index = []
    for chord in chords:
        if chord in chord_to_index:
            chords_index.append(chord_to_index[chord])
        else:
            chords_index.append(chord_to_index[UNK])
    return chords_index


def chords_to_index_save(chords_file, save_path, chord_to_index):
    chords = pickle.load(open(str(chords_file), 'rb'))
    chords_index = chords_to_index(chords, chord_to_index)  # todo, refactor, name rethink
    pickle.dump(chords_index, open(str(save_path) , 'wb'))

    
def chords_to_indexes(chords_dir, chords_index_dir):
    chord_to_index, index_to_chord = get_chord_dict()
    for chords_file in chords_dir.glob('*.pickle'):
        try:
            chords_to_index_save(chords_file, chords_index_dir / chords_file.name,
                                 chord_to_index)
        except (ValueError, OSError) as e:
            exception_str = 'Unexpected error in ' + str(midi_file)  + ':\n', e, sys.exc_info()[0]
            print(exception_str)
            

def save_index_from_chords(chords_root_dir, chords_index_root_dir):
    for chords_dir in tqdm(chords_root_dir.glob('**/*')):
        if is_containing_data_directly(chords_dir):
            chords_index_dir = Path(str(chords_dir).replace(chords_name, chords_index_name))
            if not(chords_index_dir.exists()):
                chords_index_dir.mkdir(parents=True)
            chords_to_indexes(chords_dir, chords_index_dir)
            
generate_target_from_original(original_root_dir=chords_root_dir,
                              target_root_dir=chords_index_root_dir,
                              original_name=chords_name,
                              target_name=chords_index_name,
                              original_suffix='pickle',
                              target_suffix='pickle',
                              process_file=chords_to_index_save,
                                  chord_to_index = pickle.load(open("data/chord_to_index.pickle", 'rb'))
    index_to_chord = pickle.load(open("data/index_to_chord.pickle", 'rb'))
                              )


def save_tempo_changed_midi():
    generate_target_from_original(original_root_dir=config.debug_root_dir,
                                  target_root_dir=config.tempo_root_dir,
                                  original_name=config.debug_name,
                                  target_name=config.tempo_name,
                                  original_suffix="mid",
                                  target_suffix="mid",
                                  process_file=change_tempo_midi)


def preprocess():

    save_tempo_changed_midi()




if __name__ == "__main__":
    preprocess()