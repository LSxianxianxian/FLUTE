import pytorch_lightning as pl
from typing import Iterator, Tuple, List, Optional

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.wsd.dataset import ConsecDataset, ConsecSample, ConsecDefinition
from src.wsd.tokenizer import DeBERTaTokenizer, ConsecTokenizer


def predict(
    module: pl.LightningModule,
    tokenizer: ConsecTokenizer,
    samples: Iterator[ConsecSample],
    text_encoding_strategy: str,
    token_batch_size: int = 1024,
    progress_bar: bool = False,
) -> Iterator[Tuple[ConsecSample, List[float]]]:

    # todo only works on single gpu
    device = next(module.parameters()).device

    # todo hardcoded dataset
    dataset = ConsecDataset.from_samples(
        samples,
        tokenizer=tokenizer,
        use_definition_start=True,
        text_encoding_strategy=text_encoding_strategy,
        tokens_per_batch=token_batch_size,
        max_batch_size=128,
        section_size=2_000,
        prebatch=True,
        shuffle=False,
        max_length=tokenizer.model_max_length,
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

    # predict

    iterator = dataloader
    progress_bar = tqdm() if progress_bar else None

    for batch in iterator:

        batch_samples = batch["original_sample"]
        batch_definitions_positions = batch["definitions_positions"]

        with autocast(enabled=True):
            with torch.no_grad():
                batch_out = module(
                    **{
                        k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in batch.items()
                    }
                )
                batch_predictions = batch_out["pred_probs"]

        for sample, dp, probs in zip(
            batch_samples, batch_definitions_positions, batch_predictions
        ):
            definition_probs = []
            for start in dp:
                definition_probs.append(probs[start].item())
            yield sample, definition_probs
            if progress_bar is not None:
                progress_bar.update()

    if progress_bar is not None:
        progress_bar.close()
