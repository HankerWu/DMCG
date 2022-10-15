import os
import argparse
from rdkit import Chem
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
from confgen.e2c.dataset import PygGeomDataset
from confgen.model.gnn import GNN
import pickle
from confgen.utils.utils import set_rdmol_positions
from rdkit.Chem import AllChem

def load_model(ckpt_dir, device, sample_beta=1.2):
    assert os.path.exists(ckpt_dir)
    checkpoint = torch.load(ckpt_dir, map_location=device)
    ckpt_args = checkpoint["args"]

    shared_params = {
        "mlp_hidden_size": ckpt_args.mlp_hidden_size,
        "mlp_layers": ckpt_args.mlp_layers,
        "latent_size": ckpt_args.latent_size,
        "use_layer_norm": ckpt_args.use_layer_norm,
        "num_message_passing_steps": ckpt_args.num_layers,
        "global_reducer": ckpt_args.global_reducer,
        "node_reducer": ckpt_args.node_reducer,
        "dropedge_rate": ckpt_args.dropedge_rate,
        "dropnode_rate": ckpt_args.dropnode_rate,
        "dropout": ckpt_args.dropout,
        "layernorm_before": ckpt_args.layernorm_before,
        "encoder_dropout": ckpt_args.encoder_dropout,
        "use_bn": ckpt_args.use_bn,
        "vae_beta": ckpt_args.vae_beta,
        "decoder_layers": ckpt_args.decoder_layers,
        "reuse_prior": ckpt_args.reuse_prior,
        "cycle": ckpt_args.cycle,
        "pred_pos_residual": ckpt_args.pred_pos_residual,
        "node_attn": ckpt_args.node_attn,
        "global_attn": ckpt_args.global_attn,
        "shared_decoder": ckpt_args.shared_decoder,
        "sample_beta": sample_beta,
        "shared_output": ckpt_args.shared_output,
        "not_origin": False,
    }
    model = GNN(**shared_params).to(device)
    cur_state_dict = model.state_dict()
    del_keys = []
    state_dict = checkpoint["model_state_dict"]
    for k in state_dict.keys():
        if k not in cur_state_dict:
            del_keys.append(k)
    for k in del_keys:
        del state_dict[k]
    model.load_state_dict(state_dict)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")
    
    return model

def load_dataset(mol_list, batch_size=1, num_workers=1):
    dataset = PygGeomDataset(
        root="dataset",
        inference=True,
        inference_mols=mol_list
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,)
    return dataloader

def optimize_confomer(mol):
    try:
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=3000)
        mol = Chem.RemoveHs(mol)
        return mol
    except:
        print("mmff error")
        return None

def evaluate(model, device, mol_list, repeat=1, batch_size=1, num_workers=1):
    model.eval()
    mol_preds = []
    loader = load_dataset(mol_list, batch_size, num_workers)
    for r in range(repeat):
        mol_preds.append([])
        for batch in tqdm(loader, desc="Iteration"):
            batch = batch.to(device)
        
            with torch.no_grad():
                pred, _ = model(batch, sample=True)
            pred = pred[-1]
            bsz = batch.num_graphs
            n_nodes = batch.n_nodes.tolist()
            pre_nodes = 0
            for i in range(bsz):
                mol = set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
                mol = optimize_confomer(mol)
                pre_nodes += n_nodes[i]
                mol_preds[r].append(mol)

    return mol_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--eval-from", type=str, default=None)
    parser.add_argument("--input", type=str, default="mpro_mols.txt")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--sample-beta", type=float, default=1.2)

    args = parser.parse_args()
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    
    with open(args.input, 'r') as f:
        mol_list = [Chem.MolFromSmiles(smi.strip()) for smi in f]
    
    model = load_model(args.eval_from, device, args.sample_beta)
    results = evaluate(model, device, mol_list, args.repeat, batch_size=args.batch_size, num_workers=args.num_workers)
    
    with open(f"{args.input}.pkl", "wb") as f:
        pickle.dump(results, f)
    

if __name__ == "__main__":
    main()
