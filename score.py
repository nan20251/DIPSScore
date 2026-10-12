"""Score two already-placed protein PDBs with a newly trained checkpoint."""
import argparse
import torch
from data import ca_tensors
from checkpoint import load_checkpoint

def read_pdb(path):
    from biopandas.pdb import PandasPdb
    frame = PandasPdb().read_pdb(path).df['ATOM'].rename(columns={
        'residue_name':'resname', 'residue_number':'residue', 'chain_id':'chain',
        'x_coord':'x', 'y_coord':'y', 'z_coord':'z'})
    if 'insertion' in frame and frame['insertion'].astype(str).str.strip().ne('').any():
        frame['residue'] = frame.residue.astype(str) + frame.insertion.fillna('').astype(str).str.strip()
    return ca_tensors(frame)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--receptor', required=True)
    p.add_argument('--ligand', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    from dips_score import DIPSScore
    checkpoint = load_checkpoint(a.checkpoint, a.device)
    model = DIPSScore(a.device, checkpoint.get('config', {}).get('topk',1500))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    with torch.no_grad():
        print(model(*read_pdb(a.ligand), *read_pdb(a.receptor)).item())

if __name__ == '__main__':
    main()
