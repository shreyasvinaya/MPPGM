"""Runner for molgan"""
import sys

import torch
import torch.nn.functional as F

from tqdm import tqdm

import pandas as pd
import numpy as np


from deepchem.molnet import load_tox21
from deepchem.data import NumpyDataset
from deepchem.feat import MolGanFeaturizer
from deepchem.models.torch_models import BasicMolGANModel
from deepchem.models.optimizers import ExponentialDecay

from rdkit import Chem

from eval import MolecularMetrics

torch.cuda.empty_cache()
# torch.random.manual_seed(43)

class ModelArgs:
    edges: int = 5
    vertices: int = 9
    nodes: int = 5
    embedding_dim: int = 100
    dropout_rate: float = 0.0
    device: str = torch.device('cpu')


class TrainArgs:
    learning_rate: float = 0.001
    batch_size: int = 1024
    n_epochs: int = 1000
    checkpoint_interval: int = 20
    generator_steps: float = 0.2


class Trainer:

    def __init__(self,
                 model_args: ModelArgs,
                 train_args: TrainArgs,
                 dataset: pd.DataFrame,
                 featurized: bool = False,
                 state_dict: dict = None):

        self.model = BasicMolGANModel(edges=model_args.edges,
                                      vertices=model_args.vertices,
                                      nodes=model_args.nodes,
                                      embedding_dim=model_args.embedding_dim,
                                      dropout=model_args.dropout_rate,
                                      device=model_args.device,
                                      learning_rate=train_args.learning_rate,
                                      batch_size=train_args.batch_size,
                                      model_dir="models/molgan")
        self.train_args = train_args
        self.model_args = model_args
        self.dataset = dataset
        self.eval_dataset = self.dataset
        self.featurizer = MolGanFeaturizer()
        self.smiles = list(dataset['smiles'])
        if not featurized:
            print("Featurizing dataset")
            self.dataset = self.featurizer.featurize(self.smiles)
            print("Dataset featurized")
            self.dataset = [i for i in self.dataset if type(i) != np.ndarray]
            self.dataset = [i for i in self.dataset if i is not None]

        self.dataset = NumpyDataset([x.adjacency_matrix for x in self.dataset],
                                    [x.node_features for x in self.dataset])
        # No splits in the dataset in accordance with the original implementation

    def model_init(self):
        self.reloaded_model = BasicMolGANModel(
            edges=self.model_args.edges,
            vertices=self.model_args.vertices,
            nodes=self.model_args.nodes,
            embedding_dim=self.model_args.embedding_dim,
            dropout=self.model_args.dropout_rate,
            device=self.model_args.device,
            learning_rate=self.train_args.learning_rate,
            batch_size=self.train_args.batch_size,
            model_dir="models/molgan")

    def iterbatches(self):
        best_validity = -1
        patience = 10
        i = 0
        while (i < self.train_args.n_epochs):
            print("Epoch: ", i + 1)
            #for i in tqdm(range(self.train_args.n_epochs)):
            for batch in self.dataset.iterbatches(
                    batch_size=self.train_args.batch_size, pad_batches=True):
                adjacency_tensor = F.one_hot(
                    torch.Tensor(batch[0]).to(torch.int64),
                    self.model.edges).to(torch.float32)
                node_tensor = F.one_hot(
                    torch.Tensor(batch[1]).to(torch.int64),
                    self.model.nodes).to(torch.float32)
                yield {
                    self.model.data_inputs[0]: adjacency_tensor,
                    self.model.data_inputs[1]: node_tensor
                }
            # Evaluate the model and restore if collapse
            if (
                    i + 1
            ) % self.train_args.checkpoint_interval == 0 and i + 1 > self.train_args.n_epochs * 0.2:
                validity = self.evaluate()
                if validity < best_validity * 0.95:
                    # self.model_init()
                    self.model.restore(model_dir="models/molgan2")
                    # self.model = self.reloaded_model
                    print("Model restored")
                    print("Best validity: ", best_validity)
                    patience -= 1
                else:
                    best_validity = validity
                    self.model.save_checkpoint(model_dir="models/molgan2",
                                               max_checkpoints_to_keep=1)
                    print("Model saved")
                    patience = 10

                if patience == 0:
                    print("Early stopping")
                    break
            i += 1

    def train(self):
        self.model.fit_gan(
            self.iterbatches(),
            generator_steps=self.train_args.generator_steps,
            checkpoint_interval=self.train_args.checkpoint_interval,
            max_checkpoints_to_keep=25)

    def generate(self, n_samples=10, eval=False):
        generated_data = self.model.predict_gan_generator(n_samples)
        # convert graphs to RDKitmolecules
        nmols = self.featurizer.defeaturize(generated_data)
        # print(nmols)
        # print("{} molecules generated".format(len(nmols)))
        return nmols

        nmols = list(filter(lambda x: x is not None, nmols))
        print("{} valid molecules".format(len(nmols)))

        nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
        print("Generated SMILES: ", nmols_smiles)
        return nmols_smiles

    def evaluate(self):
        validity_scores = []
        uniqueness_scores = []
        novelty_scores = []
        synthezaibility_scores = []
        drug_like_scores = []
        for _ in range(5):
            nmols_smiles = self.generate(n_samples=1000)
            validity_scores.append(
                MolecularMetrics.valid_total_score(nmols_smiles))
            uniqueness_scores.append(
                MolecularMetrics.unique_total_score(nmols_smiles))
            novelty_scores.append(
                MolecularMetrics.novel_total_score(nmols_smiles,
                                                   self.eval_dataset))
            synthezaibility_scores.append(
                MolecularMetrics.synthetic_accessibility_score_scores(
                    nmols_smiles).mean())
            drug_like_scores.append(
                MolecularMetrics.drugcandidate_scores(
                    nmols_smiles, self.eval_dataset).mean())
        sys.stdout.flush()
        print("Validity: ", sum(validity_scores) / len(validity_scores))
        print("Uniqueness: ", sum(uniqueness_scores) / len(uniqueness_scores))
        print("Novelty: ", sum(novelty_scores) / len(novelty_scores))
        print("Synthezaibility: ",
              sum(synthezaibility_scores) / len(synthezaibility_scores))
        print("Drug Likeness: ", sum(drug_like_scores) / len(drug_like_scores))

        return sum(validity_scores) / len(validity_scores)


def main():
    """_summary_
    """
    model_args = ModelArgs()
    model_args.device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('mps')
    train_args = TrainArgs()
    # featurizer = MolGanFeaturizer(
    #     max_atom_count=10, atom_labels=[0, 5, 6, 7, 8, 9, 11, 12, 13, 14])
    print("Loading QM9 dataset")
    # df = pd.DataFrame(moses.get_dataset("train"), columns=["smiles"])
    # df = pd.read_csv("moses/data/train.csv")
    # df.rename(columns={"SMILES": "smiles"}, inplace=True)
    # print(df.head())
    tasks, datasets, transformers = load_tox21()
    df = pd.DataFrame(data={'smiles': datasets[0].ids})
    df = df.sample(frac=1).reset_index(drop=True)
    # input_file = os.path.join("deepchem/deepchem/models/tests/assets/",
    #                           "molgan_example.csv")
    # df = pd.read_csv(input_file)
    dataset = df
    print("Dataset loaded")
    print(df.info())
    trainer = Trainer(model_args, train_args, dataset, featurized=False)
    trainer.train()
    print("Training complete")
    trainer.evaluate()
    print("Evaluation complete")


if __name__ == "__main__":
    main()
