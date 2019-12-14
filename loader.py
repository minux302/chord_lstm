import pickle
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
                 # batch_size,
                 # seq_len
                 ):

        self.file_list = load_chord_indexes(dataset_path)
        self.song_batch_size = song_batch_size
        # self.batch_size = batch_size
        # self.seq_len = seq_len

        self.file_index = 0

    def _get_chord_indexes(self):
        chord_indexes = pickle.load(open(self.file_list[self.file_index], 'rb'))
        self.file_index += 1
        yield chord_indexes

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

    def generate_batches(self):
        song_batch = []
        for i in range(self.song_batch_size):
            # song_batch.append(self._get_chord_indexes().__next__())
            song_batch.append(self._preprocess(self._get_chord_indexes().__next__()))
        return song_batch



if __name__ == "__main__":
    loader = ChordLoader(dataset_path='data/chord_indexes',
                         song_batch_size=16)
    print(loader.generate_batches())
    print(loader.generate_batches())