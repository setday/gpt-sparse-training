# сохраняет датасет WikiText в бинарные файлы для обучения (train.bin / val.bin / test.bin)
# сплитов у wikitext уже три: train / validation / test
# зависимости: datasets, tiktoken, numpy, tqdm

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets

# число воркеров в .map()
num_proc = 8
# число воркеров при загрузке датасета
num_proc_load_dataset = num_proc

# какой вариант WikiText использовать:
# варианты: "wikitext-2-raw-v1", "wikitext-103-raw-v1"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"

enc = tiktoken.get_encoding("gpt2")

def ensure_outdir():
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()
    return base_dir

def as_uint_dtype(encoding):
    # gpt2 max_token_value == 50256 -> помещается в uint16
    return np.uint16

def process(example):
    # encode_ordinary игнорирует спецтокены
    ids = enc.encode_ordinary(example["text"])
    # добавим токен конца текста (в gpt2 это 50256)
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}

if __name__ == "__main__":
    # загружаем WikiText
    dataset = load_dataset("wikitext", WIKITEXT_CONFIG, num_proc=num_proc_load_dataset)

    # токенизируем все сплиты
    tokenized = {}
    for split_name, dset in dataset.items():
        tokenized[split_name] = dset.map(
            process,
            remove_columns=dset.column_names,  # убираем 'text'
            desc=f"tokenizing {split_name}",
            num_proc=num_proc,
        )

    outdir = ensure_outdir()
    dtype = as_uint_dtype(enc)

    # конкатенируем все токены каждого сплита в один .bin
    for split_name, dset in tokenized.items():
        # общая длина массива токенов
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(outdir, f"{'val' if split_name == 'validation' else split_name}.bin")

        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        # число батчей для записи; не должно превышать число сэмплов
        total_samples = len(dset)
        total_batches = max(1, min(1024, total_samples))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # берём непрерывные шарды для ускорения конкатенации
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")

            arr_batch = np.concatenate(batch["ids"]) if len(batch) > 0 else np.array([], dtype=np.int64)
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()

        # простая сводка
        total_tokens = int(arr_len)
        print(f"{split_name}: wrote {filename} | tokens: {total_tokens:,}")

    # чтение потом:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
