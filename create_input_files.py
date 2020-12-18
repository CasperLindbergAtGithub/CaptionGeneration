from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path="C:/Users/Caspe/Documents/Chalmers/Master/DAT450/ShowAttendTell/dataset_flickr8k.json",
                       image_folder='C:/Users/Caspe/Documents/Chalmers/Master/DAT450/Project/data/Flicker8k_Dataset',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='C:/Users/Caspe/Documents/Chalmers/Master/DAT450/ShowAttendTell/output/',
                       max_len=50)
