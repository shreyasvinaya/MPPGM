

# Import packages
import torch
import numpy as np


from tqdm import tqdm
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions import transforms as tT

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys

import deepchem as dc
from deepchem.models.optimizers import Adam
from deepchem.data import NumpyDataset
from deepchem.splits import RandomSplitter
from deepchem.molnet import load_tox21
from torch.distributions.multivariate_normal import MultivariateNormal

import rdkit
from rdkit import Chem

import selfies as sf
from nflows import *
from eval import MolecularMetrics

# Download from MolNet
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='ECFP')
df = pd.DataFrame(data={'smiles': datasets[0].ids})
"""# Data Pre-processing

Sampling to save time
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = df[['smiles']]
"""# Featurizer"""

sf.set_semantic_constraints()  # reset constraints
constraints = sf.get_semantic_constraints()
constraints['?'] = 3

sf.set_semantic_constraints(constraints)


def preprocess_smiles(smiles):
    try:
        return sf.encoder(smiles)
    except:
        return


def keys_int(symbol_to_int):
    d = {}
    i = 0
    for key in symbol_to_int.keys():
        d[i] = key
        i += 1
    return d


data['selfies'] = data['smiles'].apply(preprocess_smiles)
data = data.dropna()

data['len'] = data['smiles'].apply(lambda x: len(x))

data.sort_values(by='len').head()

selfies_list = np.asanyarray(data.selfies)
selfies_alphabet = sf.get_alphabet_from_selfies(selfies_list)
selfies_alphabet.add(
    '[nop]')  # Add the "no operation" symbol as a padding character
selfies_alphabet.add('.')
selfies_alphabet = list(sorted(selfies_alphabet))
largest_selfie_len = max(sf.len_selfies(s) for s in selfies_list)
symbol_to_int = dict((c, i) for i, c in enumerate(selfies_alphabet))
int_mol = keys_int(symbol_to_int)
print("Largest selfie len:", largest_selfie_len)

onehots = sf.batch_selfies_to_flat_hot(selfies_list, symbol_to_int,
                                       largest_selfie_len)

input_tensor = torch.tensor(onehots)
noise_tensor = torch.rand(input_tensor.shape)
dequantized_data = torch.add(input_tensor, noise_tensor)

ds = NumpyDataset(dequantized_data)  # Create a DeepChem dataset
splitter = RandomSplitter()
train_s, val, test = splitter.train_valid_test_split(dataset=ds, seed=42)
train_idx, val_idx, test_idx = splitter.split(dataset=ds, seed=42)

dim = len(train_s.X[0])  # length of one-hot encoded vectors
print("Dim:", dim)
print("X len", train_s.X)  # 2000 samples,

K = 2
# torch.manual_seed(0)

latent_size = dim
b = torch.Tensor([1 if i % 2 == 0 else 0
                  for i in range(latent_size)]).to(device)
flows = []
for i in range(K):
    s = MLP([latent_size, 2 * latent_size, latent_size],
            init_zeros=True,
            device=device)
    t = MLP([latent_size, 2 * latent_size, latent_size],
            init_zeros=True,
            device=device)
    if i % 2 == 0:
        flows += [MaskedAffineFlow(b, t, s)]
    else:
        flows += [MaskedAffineFlow(1 - b, t, s)]
    flows += [ActNorm(latent_size)]

# Set prior and q0

distribution = MultivariateNormal(torch.zeros(dim, device=device),
                                  torch.eye(dim, device=device))

# Construct flow model
nfm = NormFlow(flows, distribution, dim)

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda')
nfm = nfm.to(device)
nfm = nfm.to(torch.float32)

nfm.parameters()
"""training params"""

# Train model
max_iter = 100

loss_hist = np.array([])

optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-4)
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()

    # Get training samples
    x = dequantized_data

    # Compute loss
    loss = nfm.log_prob(x.to(device))

    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    print("Loss: ", loss)

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()
plt.savefig('loss_tox21.png')


def evaluate(eval_dataset, nfm, n_samples=1000):
    validity_scores = []
    uniqueness_scores = []
    novelty_scores = []
    synthezaibility_scores = []
    drug_like_scores = []
    for _ in range(5):
        generated_samples, _ = nfm.sample(n_samples)
        mols = torch.floor(generated_samples)
        mols = torch.clamp(mols, 0, 1)
        mols_list = mols.cpu().detach().numpy().tolist()
        for mol in mols_list:
            for i in range(largest_selfie_len):
                row = mol[len(selfies_alphabet) * i:len(selfies_alphabet) *
                          (i + 1)]
                if all(elem == 0 for elem in row):
                    mol[len(selfies_alphabet) * (i + 1) - 1] = 1
        valid_count = 0
        valid_selfies, invalid_selfies = [], []
        for idx, selfies in enumerate(mols):
            try:
                if Chem.MolFromSmiles(sf.decoder(mols[idx]),
                                      sanitize=True) is not None:
                    valid_count += 1
                    valid_selfies.append(selfies)
                else:
                    invalid_selfies.append(selfies)
            except Exception:
                pass
        print("number of valid molecules:", valid_count)
        nmols_smiles = [
            Chem.MolFromSmiles(sf.decoder(vs)) for vs in valid_selfies
        ]
        print(nmols_smiles)
        validity_scores.append(len(valid_selfies) / n_samples)
        uniqueness_scores.append(
            MolecularMetrics.unique_total_score(nmols_smiles))
        novelty_scores.append(
            MolecularMetrics.novel_total_score(nmols_smiles, eval_dataset))
        synthezaibility_scores.append(
            MolecularMetrics.synthetic_accessibility_score_scores(
                nmols_smiles).mean())
        drug_like_scores.append(
            MolecularMetrics.drugcandidate_scores(nmols_smiles,
                                                  eval_dataset).mean())
    sys.stdout.flush()
    print("Validity: ", validity_scores)
    print("Validity: ", sum(validity_scores) / len(validity_scores))
    print("Uniqueness: ", sum(uniqueness_scores) / len(uniqueness_scores))
    print("Novelty: ", sum(novelty_scores) / len(novelty_scores))
    print("Synthezaibility: ",
          sum(synthezaibility_scores) / len(synthezaibility_scores))
    print("Drug Likeness: ", sum(drug_like_scores) / len(drug_like_scores))

    return sum(validity_scores) / len(validity_scores)

    for _ in range(5):

        # Add padding characters if needed
        for mol in mols_list:
            for i in range(self.largest_selfie_len):
                row = mol[len(self.selfies_alphabet) *
                          i:len(self.selfies_alphabet) * (i + 1)]
                if all(elem == 0 for elem in row):
                    mol[len(self.selfies_alphabet) * (i + 1) - 1] = 1
        valid_count = 0
        valid_selfies, invalid_selfies = [], []
        for idx, selfies in enumerate(mols):
            try:
                if Chem.MolFromSmiles(sf.decoder(mols[idx]),
                                      sanitize=True) is not None:
                    valid_count += 1
                    valid_selfies.append(selfies)
                else:
                    invalid_selfies.append(selfies)
            except Exception:
                pass
        nmols_smiles = [sf.decoder(vs) for vs in valid_selfies]
        print(nmols_smiles)
        validity_scores.append(len(valid_selfies) / n_samples)
        uniqueness_scores.append(
            MolecularMetrics.unique_total_score(nmols_smiles))
        novelty_scores.append(
            MolecularMetrics.novel_total_score(nmols_smiles, eval_dataset))
        synthezaibility_scores.append(
            MolecularMetrics.synthetic_accessibility_score_scores(
                nmols_smiles).mean())
        drug_like_scores.append(
            MolecularMetrics.drugcandidate_scores(nmols_smiles,
                                                  eval_dataset).mean())
    sys.stdout.flush()
    print("Validity: ", sum(validity_scores) / len(validity_scores))
    print("Uniqueness: ", sum(uniqueness_scores) / len(uniqueness_scores))
    print("Novelty: ", sum(novelty_scores) / len(novelty_scores))
    print("Synthezaibility: ",
          sum(synthezaibility_scores) / len(synthezaibility_scores))
    print("Drug Likeness: ", sum(drug_like_scores) / len(drug_like_scores))


from rdkit import RDLogger
from rdkit import Chem

RDLogger.DisableLog('rdApp.*')  # suppress error messages
"""# Generate Primitive"""
validity_scores = []
uniqueness_scores = []
novelty_scores = []
synthezaibility_scores = []
drug_like_scores = []
n_samples = 1000
eval_dataset = df
for _ in range(5):
    generated_samples, _ = nfm.sample(1000)
    mols = torch.floor(generated_samples)
    mols = torch.clamp(mols, 0, 1)
    mols_list = mols.cpu().detach().numpy().tolist()
    # Add padding characters if needed
    for mol in mols_list:
        for i in range(largest_selfie_len):
            row = mol[len(selfies_alphabet) * i:len(selfies_alphabet) *
                      (i + 1)]
            if all(elem == 0 for elem in row):
                mol[len(selfies_alphabet) * (i + 1) - 1] = 1

    mols = sf.batch_flat_hot_to_selfies(mols_list, int_mol)

    valid_count = 0
    valid_selfies, invalid_selfies = [], []
    for idx, selfies in enumerate(mols):
        try:
            if Chem.MolFromSmiles(sf.decoder(mols[idx]),
                                  sanitize=True) is not None:
                valid_count += 1
                valid_selfies.append(selfies)
            else:
                invalid_selfies.append(selfies)
        except Exception:
            pass
    print("number of valid molecules:", valid_count)
    print("number of len(mols)", len(mols))
    gen_smiles = [Chem.MolFromSmiles(sf.decoder(vs)) for vs in valid_selfies]
    nmols_smiles = gen_smiles
    validity_scores.append(len(valid_selfies) / n_samples)
    uniqueness_scores.append(MolecularMetrics.unique_total_score(nmols_smiles))
    novelty_scores.append(
        MolecularMetrics.novel_total_score(nmols_smiles, eval_dataset))
    synthezaibility_scores.append(
        MolecularMetrics.synthetic_accessibility_score_scores(
            nmols_smiles).mean())
    drug_like_scores.append(
        MolecularMetrics.drugcandidate_scores(nmols_smiles,
                                              eval_dataset).mean())
sys.stdout.flush()
print("Validity: ", sum(validity_scores) / len(validity_scores))
print("Uniqueness: ", sum(uniqueness_scores) / len(uniqueness_scores))
print("Novelty: ", sum(novelty_scores) / len(novelty_scores))
print("Synthezaibility: ",
      sum(synthezaibility_scores) / len(synthezaibility_scores))
print("Drug Likeness: ", sum(drug_like_scores) / len(drug_like_scores))
# print("Valid Count: ", valid_count)
"""# Evaluate"""
# Evaluate the generated molecules
# evaluate(train_s, nfm, n_samples=1000)
