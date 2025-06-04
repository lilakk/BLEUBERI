from dataset import KeywordDataset

def create_dataset(
    data_path, 
    split="train", 
    cache_dir="",
    streaming=False,
    shuffle=False, 
    load_from_disk=False,
    tokenizer=None,
):
    ds = KeywordDataset(data_path, tokenizer)

    ds.load_dataset(
        split, 
        cache_dir,
        streaming=streaming,
        shuffle=shuffle,
        load_from_disk=load_from_disk
    )

    return ds
