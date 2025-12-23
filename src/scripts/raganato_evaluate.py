import tempfile
from typing import Tuple, Any, Dict, Optional, List

import torch

from src.wsd.dataset import ConsecSample
from src.wsd.dependency_finder import EmptyDependencyFinder
from src.pl_model_modules import ModelModule
from src.wsd.continuous_predict import Predictor
from src.wsd.sense_inventories import SenseInventory, WordNetSenseInventory
from src.utils.commons import execute_bash_command
from src.utils.wsd import expand_raganato_path

import conf
from hparams import Hparams


def framework_evaluate(framework_folder: str, gold_file_path: str, pred_file_path: str) -> Tuple[float, float, float]:
    scorer_folder = f"{framework_folder}/Evaluation_Datasets"
    command_output = execute_bash_command(
        f"[ ! -e {scorer_folder}/Scorer.class ] && javac -d {scorer_folder} {scorer_folder}/Scorer.java; java -cp {scorer_folder} Scorer {gold_file_path} {pred_file_path}"
    )
    command_output = command_output.split("\n")
    p, r, f1 = [float(command_output[i].split("=")[-1].strip()[:-1]) for i in range(3)]
    return p, r, f1


def sample_prediction2sense(sample: ConsecSample, prediction: int, sense_inventory: SenseInventory) -> str:
    sample_senses = sense_inventory.get_possible_senses(sample.disambiguation_instance.lemma, sample.disambiguation_instance.pos)
    sample_definitions = [sense_inventory.get_definition(s) for s in sample_senses]

    for s, (d, ri) in zip(sample_senses, sample_definitions):
        if d == sample.candidate_definitions[prediction].text:
            return s

    raise ValueError


def raganato_evaluate(
    raganato_path: str,
    wsd_framework_dir: str,
    module: ModelModule,
    predictor: Predictor,
    wordnet_sense_inventory: WordNetSenseInventory,
    samples_generator,
    prediction_params: Dict[Any, Any],
    fine_grained_evals: Optional[List[str]] = None,
    reporting_folder: Optional[str] = None,
) -> Tuple[float, float, float, Optional[List[Tuple[str, float, float, float]]]]:

    # load tokenizer
    tokenizer = conf.TokenizerInst.instantiate()

    # instantiate samples
    consec_samples = list(samples_generator())

    # predict
    disambiguated_samples = predictor.predict(
        consec_samples,
        already_kwown_predictions=None,
        reporting_folder=reporting_folder,
        wordnet_sense_inventory=wordnet_sense_inventory,
        **dict(module=module, tokenizer=tokenizer, **prediction_params),
    )

    # write predictions and evaluate
    fge_scores = []

    # write predictions and evaluate
    with tempfile.TemporaryDirectory() as tmp_dir:

        # write predictions to tmp file
        with open(f"{tmp_dir}/predictions.gold.key.txt", "w") as f:
            for sample, idx in disambiguated_samples:
                f.write(f"{sample.sample_id} {sample_prediction2sense(sample, idx, wordnet_sense_inventory)}\n")

        # NVAR
        for fge in ["n", "v", "a", "r"]:
            gold_path = f"{tmp_dir}/{fge}.gold.key.txt"
            pred_path = f"{tmp_dir}/{fge}.predictions.gold.key.txt"
            with open(pred_path, "w") as f, open(gold_path, "w") as gf:
                for sample, idx in disambiguated_samples:
                    pos = sample.disambiguation_instance.pos
                    pos = pos if pos != "s" else "a"
                    if pos == fge:
                        f.write(f"{sample.sample_id} {sample_prediction2sense(sample, idx, wordnet_sense_inventory)}\n")
                        g_labels = " ".join(sample.disambiguation_instance.labels)
                        gf.write(f"{sample.sample_id} {g_labels}\n")
            _p, _r, _f1 = framework_evaluate(
                wsd_framework_dir,
                gold_file_path=gold_path,
                pred_file_path=pred_path,
            )
            fge_scores.append((fge, _p, _r, _f1))

        # se2, se3, se13, se15
        for fge in [
            "senseval2",
            "senseval3",
            "semeval2007",
            "semeval2013",
            "semeval2015",
        ]:
            gold_path = f"{tmp_dir}/{fge}.gold.key.txt"
            pred_path = f"{tmp_dir}/{fge}.predictions.gold.key.txt"
            with open(pred_path, "w") as f, open(gold_path, "w") as gf:
                for sample, idx in disambiguated_samples:
                    sample_id = sample.sample_id
                    if len(sample_id.split(".")) > 3:
                        corpus_name = sample_id.split(".")[0]
                        if corpus_name == fge:
                            f.write(f"{sample.sample_id} {sample_prediction2sense(sample, idx, wordnet_sense_inventory)}\n")
                            g_labels = " ".join(sample.disambiguation_instance.labels)
                            gf.write(f"{sample.sample_id} {g_labels}\n")
            _p, _r, _f1 = framework_evaluate(
                wsd_framework_dir,
                gold_file_path=gold_path,
                pred_file_path=pred_path,
            )
            fge_scores.append((fge, _p, _r, _f1))

        # compute metrics
        p, r, f1 = framework_evaluate(
            wsd_framework_dir,
            gold_file_path=expand_raganato_path(raganato_path)[1],
            pred_file_path=f"{tmp_dir}/predictions.gold.key.txt",
        )

        return p, r, f1, fge_scores


def main() -> None:

    # load module
    # todo decouple ConsecPLModule
    hps = Hparams()
    module = ModelModule.load_from_checkpoint(conf.path_checkpoint_4_test, hps=hps)
    if conf.gpus is not None:
        device = "cuda:0"
    else:
        device = "cpu"
    module.to(torch.device(device))
    module.eval()
    module.freeze()
    module.sense_extractor.evaluation_mode = True  # no loss will be computed even if labels are passed

    # instantiate sense inventory
    sense_inventory = conf.InventoryInst.instantiate()

    # instantiate predictor
    predictor = conf.GreedyDepPredictorInst.instantiate()

    sample_generator = conf.EvalSampleGeneratorInst4RaganatoEvaluateCallBackInst.instantiate()
    test_sample_generator = conf.TestSampleGeneratorInst4RaganatoEvaluateCallBackInst.instantiate()

    # evaluate
    p, r, f1, fge_scores = raganato_evaluate(
        raganato_path=conf.path_raganato_validations[0],
        wsd_framework_dir=conf.path_wsd_framework_dir,
        module=module,
        predictor=predictor,
        wordnet_sense_inventory=sense_inventory,
        samples_generator=sample_generator,
        prediction_params={
            "text_encoding_strategy": "relative-positions",
            "token_batch_size": 4096,
            "progress_bar": False,
        },
        fine_grained_evals=[],
        reporting_folder=".",  # hydra will handle it
    )
    print(f"# val_p: {p}")
    print(f"# val_r: {r}")
    print(f"# val_f1: {f1}")

    if fge_scores:
        for fge, p, r, f1 in fge_scores:
            print(f"# {fge}: ({p:.1f}, {r:.1f}, {f1:.1f})")

    # evaluate
    test_p, test_r, test_f1, test_fge_scores = raganato_evaluate(
        raganato_path=conf.path_raganato_tests[0],
        wsd_framework_dir=conf.path_wsd_framework_dir,
        module=module,
        predictor=predictor,
        wordnet_sense_inventory=sense_inventory,
        samples_generator=test_sample_generator,
        prediction_params={
            "text_encoding_strategy": "relative-positions",
            "token_batch_size": 4096,
            "progress_bar": False,
        },
        fine_grained_evals=[],
        reporting_folder=".",  # hydra will handle it
    )
    print(f"# test_p: {test_p}")
    print(f"# test_r: {test_r}")
    print(f"# test_f1: {test_f1}")

    if test_fge_scores:
        for fge, p, r, f1 in test_fge_scores:
            print(f"# {fge}: ({p:.1f}, {r:.1f}, {f1:.1f})")
