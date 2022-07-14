import ltn
from ltn import Predicate
import torch
from torch import nn


class BasicModule(nn.Module):
    # A basic module, based on which different predicates are built.
    def __init__(self, input_size, hidden_size, output_size):
        super(BasicModule, self).__init__()
        # NN，两个线性层拟合predicate
        self.basic_module = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        ])

    def forward(self, focal, other):
        focal = focal.repeat(1, other.shape[1], 1)
        x = torch.cat([focal, other], dim=-1)
        x = self.basic_module[0](x)
        x = self.basic_module[1](x)
        x = self.basic_module[2](x)
        return self.basic_module[3](x)


# Basic Connectives
# https://github.com/tommasocarraro/LTNtorch/blob/main/tutorials/3-knowledgebase-and-learning.ipynb [in 6]
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=6), quantifier="e")


# Basic Predicates
def ltn_influence(torch_model):
    return ltn.Predicate(model=torch_model)


def ltn_yield(torch_model):
    return ltn.Predicate(model=torch_model)


def ltn_pass(torch_model):
    return ltn.Predicate(model=torch_model)


def ltn_follow(torch_model):
    return ltn.Predicate(model=torch_model)


ltn_ishuman = ltn.Predicate(func=lambda x: torch.tensor((x == "pedestrian")+0))


# LTN traffic rules
def yield_human(other: torch.Tensor, focal: torch.Tensor,
                influence: Predicate, yield_: Predicate) -> torch.Tensor:
    focal = ltn.Variable("focal", focal)
    other = ltn.Variable("other", other)
    return Forall(other, Implies(And(influence(focal, other), ltn_ishuman(other)), yield_(focal, other)))
