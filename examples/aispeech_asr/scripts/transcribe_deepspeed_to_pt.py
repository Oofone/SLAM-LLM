import torch
import  sys
import os

in_path = sys.argv[1]
out_path = sys.argv[2]

os.makedirs(out_path, exist_ok=True)

weight_dict = torch.load(in_path)["module"]
torch.save(weight_dict, f"{out_path}/model.pt")
print("[Finish]")