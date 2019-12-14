import pickle
import numpy as np
from pathlib import Path
from data_preprocess import is_containing_data_directly


def load_chord_indexes(chord_indexes_root_dir):
    chord_indexes_root_dir = Path(chord_indexes_root_dir)
    file_list = []
    for chord_indexes_dir in chord_indexes_root_dir.glob('**/*'):
        if is_containing_data_directly(chord_indexes_dir):
            for chord_indexes_file in chord_indexes_dir.glob('*.pickle'):
                file_list.append(str(chord_indexes_file))
    return file_list


class ChordLoader:

    def __init__(self,
                 dataset_path,
                 song_batch_size,
                 batch_size,
                 seq_len,
                 loader_type,
                 split_ratio=0.8
                 ):

        assert loader_type in ["train", "validation"], "loader_type is train or validation"
        file_list = load_chord_indexes(dataset_path)
        if loader_type == 'train':
            self.file_list = file_list[:int(len(file_list)*split_ratio)]
            # self.file_list = file_list[:754]
            print(len(self.file_list))
        else:
            self.file_list = file_list[int(len(file_list)*split_ratio):]
            # self.file_list = file_list[754:]

        self.song_batch_size = song_batch_size
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.file_index = 0
        self.batch_index = 0
        self.input_buffer = []
        self.target_buffer = []

    def _get_chord_indexes(self):
        chord_indexes = pickle.load(open(self.file_list[self.file_index], 'rb'))
        self.file_index += 1
        return chord_indexes

    # rethink for using this fuction
    def _preprocess(self, chord_indexes):
        max_num = 3
        processed_chord_indexes = []
        before_chord = None
        counter = 0

        for chord_index in chord_indexes:
            if chord_index == before_chord:
                counter += 1
            else:
                counter = 0
            if counter < max_num:
                processed_chord_indexes.append(chord_index)
            before_chord = chord_index
        return processed_chord_indexes

    def _get_input_and_target(self, chord_indexes_batch):
        for chord_index in chord_indexes_batch:
            for i in range(len(chord_index) - self.seq_len - 1):
                self.input_buffer.append(chord_index[i:i+self.seq_len])
                self.target_buffer.append(chord_index[i+self.seq_len])

    def generate_batches(self):
        # todo
        self.input_buffer = []
        self.target_buffer = []
        self.batch_index = 0

        chord_indexes_batch = []
        for i in range(self.song_batch_size):
            # song_batch.append(self._get_chord_indexes().__next__())
            chord_indexes_batch.append(self._preprocess(self._get_chord_indexes()))
        self._get_input_and_target(chord_indexes_batch)

    # Todo use shuffle
    def get_batch(self):
        start = self.batch_index
        end = self.batch_index + self.batch_size
        self.batch_index += self.batch_size
        return np.array(self.input_buffer[start:end]), np.array(self.target_buffer[start:end])  # Todo, need to use numpy ?

    def get_song_batch_num(self):
        return int(len(self.file_list) / self.song_batch_size)

    def get_batch_num(self):
        return int(len(self.target_buffer) / self.batch_size)

    def get_total_songs(self):
        return len(self.file_list)


if __name__ == "__main__":
    loader = ChordLoader(dataset_path='data/chord_indexes',
                         song_batch_size=16,
                         batch_size=5,
                         seq_len=20,
                         loader_type='train')
    loader.generate_batches()
    input_buffer, target_buffer = loader.get_batch()
    print(input_buffer)
    print(target_buffer)

    input_buffer, target_buffer = loader.get_batch()
    print(input_buffer)
    print(target_buffer)