import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch = {}

    batch["image"] = torch.stack([item["image"] for item in dataset_items]).to(torch.float32) / 255.0
    batch["image_path"] = [item["image_path"] for item in dataset_items]

    batch["target"] = torch.tensor([item["target"] for item in dataset_items])

    return batch
