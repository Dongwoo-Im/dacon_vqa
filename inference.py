import argparse
import pandas as pd
import os.path as osp
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader

from transformers.trainer_utils import seed_worker
from transformers import BlipConfig, BlipProcessor, BlipImageProcessor, BertTokenizerFast, BlipForQuestionAnswering

from dataset import VQADataset
import utils

import warnings
warnings.filterwarnings("ignore")


@torch.no_grad()
def inference(model, loader, processor, device):
    model.eval()

    preds = []
    
    for inputs in tqdm(loader, total=len(loader)):
        for k in inputs.keys():
            inputs[k] = inputs[k].to(device)
        outputs = model.generate(**inputs)
        pred = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        preds.extend(pred)

    return preds


def main(args):
    utils.set_seeds(args.seed)

    args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Data
    test_df = pd.read_csv('data/test.csv')

    # Model
    image_processor = BlipImageProcessor()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    processor = BlipProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model_config = BlipConfig()
    model = BlipForQuestionAnswering(model_config).to(args.device)
    
    # Load weight
    trained_weight_path = osp.join('work_dirs', args.weight)
    trained_weight = torch.load(trained_weight_path, map_location='cpu')
    model.load_state_dict(trained_weight)

    # Dataset & Dataloader
    loader_dict = {"pin_memory": True, "num_workers": 4, "worker_init_fn": seed_worker}
    test_dataset = VQADataset(test_df, processor, mode='test')
    test_loader = DataLoader(test_dataset, **loader_dict)

    # inference
    preds = inference(model, test_loader, processor, args.device)

    # submission
    submission = pd.read_csv('data/sample_submission.csv')
    submission['answer'] = preds
    file_path = osp.join('submission', f"{args.weight.replace('/', '_').replace('pt','csv')}")
    if osp.exists(file_path):
        file_path = osp.splitext(file_path)[0] + '_dup.csv'
    submission.to_csv(file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight', type=str)
    args = parser.parse_args()
    main(args)
