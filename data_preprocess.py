import pickle
from collections import Counter
from pathlib import Path

import mido
import numpy as np
import pretty_midi as pm
from tqdm import tqdm

import config


def is_containing_data_directly(data_dir):
    """Return boolian of which data is directly under data_dir
       for Lakh data dir format.
    """
    return data_dir.is_dir() and len(data_dir.name) != 1


def process_dir(original_dir, target_dir, original_suffix, target_suffix,
                process_file, **kwargs):
    for original_file in original_dir.glob('*.' + original_suffix):
        try:
            process_file(original_file,
                         target_dir / original_file.name.replace(original_suffix,
                                                                 target_suffix),
                         **kwargs)
        except (ValueError, OSError) as e:
            print('Unexpected error in ' + str(original_file), e)


def generate_target_from_original(original_root_dir, target_root_dir,
                                  original_suffix, target_suffix,
                                  process_file, **kwargs):
    """Util func for processing original data to target data.
        Args:
            original_root_dir (pathlib object): Path to root dir of Lakh dir
                format, like `lmd_matched`.
            target_root_dir (pathlib objext): Path to root dir of Lakh dir
                format. Usually it is empty before working this func.
            original_suffix (str): Data suffix before it is processed.
            target_suffix (str): Data suffix after it is processed.
            process_file (func): Func for processing each data.
            (**kwargs: Optional args for process_file())
    """
    for original_dir in tqdm(original_root_dir.glob('**/*')):
        if is_containing_data_directly(original_dir):
            target_dir = Path(str(original_dir).replace(original_root_dir.name,
                                                        target_root_dir.name))
            if not(target_dir.exists()):
                target_dir.mkdir(parents=True)
            process_dir(original_dir, target_dir,
                        original_suffix, target_suffix, process_file, **kwargs)


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


def pianoroll_to_histo(pianoroll, bar_len=config.samples_per_bar):
    bar_num = pianoroll.shape[1] // bar_len
    histo_over_octave = np.zeros((pianoroll.shape[0], bar_num))
    for i in range(bar_num):
        histo_over_octave[:, i] = np.sum(pianoroll[:, i*bar_len:(i+1)*bar_len],
                                         axis=1)
    return histo_over_octave


def compress_octave_notes(histo_over_octave, octave=config.octave):
    histo = np.zeros((octave, histo_over_octave.shape[1]))
    octave_num = histo_over_octave.shape[0] // octave
    for i in range(octave_num - 1):
        histo = np.add(histo, histo_over_octave[i*octave:(i+1)*octave])
    return histo


def midi_to_histo(midi_file, save_path):
    """Generate histogram (histo) from midi_file.
        histo (np.array):
            histo[i][j]: non zero num per bar_j for key_i
            key_i: 0-11, it means C, Db, D, ..., Bb, B.
            bar_j: index in range(time length in song // time length in bar).
    """
    mid = pm.PrettyMIDI(str(midi_file))
    pianoroll = mid.get_piano_roll()
    histo_over_octave = pianoroll_to_histo(pianoroll)  # shape: (128, pianoroll.shape[1] // bar_len)
    histo = compress_octave_notes(histo_over_octave)  # shape: (12, pianoroll.shape[1] // bar_len)
    pickle.dump(histo, open(str(save_path), 'wb'))


def histo_to_entire_histo(histo_file, save_path):
    histo = pickle.load(open(str(histo_file), 'rb'))
    entire_histo = np.sum(histo, axis=1)
    pickle.dump(entire_histo, open(save_path, 'wb'))


def note_to_index(midi_file, save_path):
    mid = pm.PrettyMIDI(str(midi_file))
    pianoroll = mid.get_piano_roll(fs=config.fs)
    index_roll = [[note_i for note_i, value in enumerate(pianoroll[:, time_i])
                   if value != 0] for time_i in range(pianoroll.shape[1])]
    pickle.dump(index_roll, open(str(save_path), 'wb'))


def histo_to_chords(histo_file, save_path):
    histo = pickle.load(open(histo_file, 'rb'))
    sorted_note_per_time = histo.argsort(axis=0)[-config.num_notes_in_chord:]
    chords = [tuple(sorted([note for note in sorted_note_per_time[:, i]]))
              for i in range(sorted_note_per_time.shape[1])]
    pickle.dump(chords, open(str(save_path), 'wb'))


def count_chords(chords_root_dir):
    chord_cntr = Counter()
    for chords_dir in tqdm(chords_root_dir.glob('**/*')):
        if is_containing_data_directly(chords_dir):
            for chords_file in chords_dir.glob('*.pickle'):
                chords = pickle.load(open(str(chords_file), 'rb'))
                for chord in chords:
                    if chord in chord_cntr:
                        chord_cntr[chord] += 1
                    else:
                        chord_cntr[chord] = 1
    return chord_cntr.most_common(n=config.num_chords - 1)


def _save_chord_dict(chords_root_dir, save_dir):
    cntr = count_chords(chords_root_dir)
    chord_to_index = dict()
    chord_to_index[config.UNK] = 0

    for chord, _ in cntr:
        chord_to_index[chord] = len(chord_to_index)  # todo refactor
    index_to_chord = {v: k for k, v in chord_to_index.items()}

    if not(save_dir.exists()):
        save_dir.mkdir(parents=True)
    pickle.dump(chord_to_index,
                open(str(save_dir / "chord_to_index.pickle"), 'wb'))
    pickle.dump(index_to_chord,
                open(str(save_dir / "index_to_chord.pickle"), 'wb'))


def chords_to_indexs(chords, chord_to_index):
    chords_index = []
    for chord in chords:
        if chord in chord_to_index:
            chords_index.append(chord_to_index[chord])
        else:
            chords_index.append(chord_to_index[config.UNK])
    return chords_index


def chords_to_index_save(chords_file, save_path, chord_to_index):
    chords = pickle.load(open(str(chords_file), 'rb'))
    chords_index = chords_to_indexs(chords, chord_to_index)  # todo, refactor, name rethink
    pickle.dump(chords_index, open(str(save_path), 'wb'))


def save_tempo_changed_midi():
    generate_target_from_original(original_root_dir=config.original_root_dir,
                                  target_root_dir=config.tempo_root_dir,
                                  original_suffix="mid",
                                  target_suffix="mid",
                                  process_file=change_tempo_midi)


def save_histogram_of_midi():
    generate_target_from_original(original_root_dir=config.tempo_root_dir,
                                  target_root_dir=config.histo_root_dir,
                                  original_suffix='mid',
                                  target_suffix='pickle',
                                  process_file=midi_to_histo)


def save_entire_histogram():
    generate_target_from_original(original_root_dir=config.histo_root_dir,
                                  target_root_dir=config.entire_histo_root_dir,
                                  original_suffix='pickle',
                                  target_suffix='pickle',
                                  process_file=histo_to_entire_histo)


def save_index_roll():
    generate_target_from_original(original_root_dir=config.tempo_root_dir,
                                  target_root_dir=config.index_roll_root_dir,
                                  original_suffix='mid',
                                  target_suffix='pickle',
                                  process_file=note_to_index)


def save_chords():
    generate_target_from_original(original_root_dir=config.histo_root_dir,
                                  target_root_dir=config.chords_root_dir,
                                  original_suffix='pickle',
                                  target_suffix='pickle',
                                  process_file=histo_to_chords)


def save_chord_dict():
    _save_chord_dict(chords_root_dir=config.chords_root_dir,
                     save_dir=config.chord_dict_dir)


def save_chords_indexes():
    generate_target_from_original(original_root_dir=config.chords_root_dir,
                                  target_root_dir=config.chords_index_root_dir,
                                  original_suffix='pickle',
                                  target_suffix='pickle',
                                  process_file=chords_to_index_save,
                                  chord_to_index = pickle.load(open("data/chord_dict/chord_to_index.pickle", 'rb'))
    )


def preprocess():

    # save_tempo_changed_midi()
    # save_histogram_of_midi()
    # save_entire_histogram()
    # save_index_roll()
    # save_chords()
    # save_chord_dict()
    save_chords_indexes()


if __name__ == "__main__":
    preprocess()