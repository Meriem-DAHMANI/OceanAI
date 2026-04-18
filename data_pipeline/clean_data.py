import re

def clean_text(text: str) -> str:
    text = re.sub(r'\{[^}]*\}', '', text)      # remove {{...}} templates
    text = re.sub(r'<[^>]*>', '', text)        # remove <...> tags
    text = re.sub(r'\[\[[^\]|]*\|([^\]]+)\]\]', r'\1', text)  # [[Page|Text]] → Text
    text = re.sub(r'\[\[([^\]|]+)\]\]', r'\1', text)          # [[Page]] → Page
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def data_split(dataset):
    # Split global train/test
    dataset_split = dataset.train_test_split(test_size=0.2, seed=42)

    train_data = dataset_split["train"]
    test_data  = dataset_split["test"]

    #  split CPT / SFT in each
    split_train = int(len(train_data) * 0.7)
    pretrain_train = train_data.select(range(0, split_train))
    finetune_train = train_data.select(range(split_train, len(train_data)))

    split_test = int(len(test_data) * 0.7)
    pretrain_test = test_data.select(range(0, split_test))
    finetune_test = test_data.select(range(split_test, len(test_data)))

    print(f"pretrain_train: {len(pretrain_train)}")
    print(f"pretrain_test:  {len(pretrain_test)}")
    print(f"finetune_train: {len(finetune_train)}")
    print(f"finetune_test:  {len(finetune_test)}")

    return pretrain_train, pretrain_test, finetune_train, finetune_test
