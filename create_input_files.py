from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)

    create_input_files(dataset='flickr8k',
                       karpathy_json_path="data/Flicker8k_Dataset/karpathy_flickr8k.json",
                       image_folder='data/Flicker8k_Dataset/Flicker8k_images/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='data/',
                       max_len=50,
                       bert_model_name='bert-base-cased')
