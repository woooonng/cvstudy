import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import json
from omegaconf import OmegaConf
from utils import choose_model
from dataset.factory import create_dataset, creat_dataloader
from metrics import top1_accuracy, f1_score

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    testset = create_dataset(cfg.DATASET.datadir, cfg.DATASET.split, cfg.MODEL.name, cfg.DATASET.transform)
    test_loader = creat_dataloader(testset, cfg.DATALOADER.batch_size, cfg.DATALOADER.shuffle)

    base_path = "results"
    models_path = os.path.join(base_path, cfg.MODEL.name)
    saved_models = [os.path.join(models_path, f) for f in os.listdir(models_path) if f.endswith(".pt")]
    
    kwargs = {"pyramid": cfg.MODEL.pyramid} if cfg.MODEL.name == "resnet50_pretrained" else {}
    model = choose_model(cfg.MODEL.name, **kwargs).to(device)

    result_file = cfg.MODEL.name + "_" + "infer_results"
    saved_file_path = os.path.join(os.path.dirname(__file__), result_file)
    results = {}
    for saved in saved_models:
        if "pyramid" in kwargs:
            if kwargs["pyramid"] == True and "pyramid" not in saved:
                continue
            elif kwargs["pyramid"] == False and "pyramid" in saved:
                continue
        weights = torch.load(saved)
        model.load_state_dict(weights)
        model_name = saved.split('/')[-1]

        all_preds = []
        all_targets = []
        model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                pred = model(inputs)

                all_preds.append(pred.argmax(dim=1).cpu())
                all_targets.append(targets.cpu())
            
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

        top1_acc = top1_accuracy(all_preds, all_targets, num_classes=10, average="macro")
        f1_macro = f1_score(all_preds, all_targets, num_classes=10, average='macro')
        results[model_name] = {
            "top1_acc": round(top1_acc, 3),
            "f1_macro": round(f1_macro, 3)
        }

    with open(saved_file_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    args = OmegaConf.from_cli()

    # load default config
    cfg = OmegaConf.load(args.configs)
    del args['configs']
    main()