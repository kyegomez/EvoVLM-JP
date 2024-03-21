import torch

import torch 
from torch import nn, Tensor
from typing import List


class BasicAverageMerge(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],  
    ):
        super(BasicAverageMerge, self).__init__()
        self.models = models
    
    def forward(self, x: Tensor) -> Tensor:
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)
    
    
    
class TaskArithmetic(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super(TaskArithmetic, self).__init__()
        self.models = nn.ModuleList(models)
        self.task_vectors = self.build_task_vectors()

    def build_task_vectors(self):
        task_vectors = []
        pretrained_params = dict(self.models[0].named_parameters())

        for model in self.models[1:]:
            task_vector = {}
            for name, param in model.named_parameters():
                if name in pretrained_params:
                    task_vector[name] = param.data - pretrained_params[name].data
            task_vectors.append(task_vector)

        return task_vectors

    def apply_task_vector(self, model, task_vector):
        for name, param in model.named_parameters():
            if name in task_vector:
                param.data += task_vector[name]

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for i, model in enumerate(self.models[1:]):
            self.apply_task_vector(model, self.task_vectors[i])
            outputs.append(model(x))

        return torch.mean(torch.stack(outputs), dim=0)
    
    

class TIESMerging:
    def __init__(self, models):
        """
        Initializes the TIESMerging class with a list of models to be merged.
        """
        self.models = models

    def reset_minimal_changes(self, threshold=0.01):
        """
        Resets parameters with minimal changes, defined by a threshold.
        """
        for model in self.models:
            for param in model.parameters():
                with torch.no_grad():
                    minimal_change_mask = torch.abs(param) < threshold
                    param[minimal_change_mask] = 0.0

    def resolve_sign_conflicts(self):
        """
        Resolves conflicting parameter signs across models by averaging.
        """
        avg_parameters = []
        for i, param in enumerate(self.models[0].parameters()):
            stacked_params = torch.stack([model.parameters()[i].data for model in self.models])
            sign_consensus = torch.sign(torch.mean(stacked_params, dim=0))
            resolved_params = torch.mean(torch.abs(stacked_params), dim=0) * sign_consensus
            avg_parameters.append(resolved_params)
        
        for model in self.models:
            for i, param in enumerate(model.parameters()):
                with torch.no_grad():
                    param.data = avg_parameters[i]

    def merge_aligned_parameters(self):
        """
        Merges only aligned parameters, assuming parameters are aligned if their product is positive.
        """
        aligned_parameters = []
        for i, param in enumerate(self.models[0].parameters()):
            stacked_params = torch.stack([model.parameters()[i].data for model in self.models])
            alignment_mask = torch.prod(stacked_params, dim=0) > 0
            aligned_param = torch.mean(stacked_params, dim=0) * alignment_mask.float()
            aligned_parameters.append(aligned_param)
        
        for model in self.models:
            for i, param in enumerate(model.parameters()):
                with torch.no_grad():
                    param.data = aligned_parameters[i]

    def merge_models(self):
        """
        Merges models using the TIES-Merging method.
        """
        self.reset_minimal_changes()
        self.resolve_sign_conflicts()
        self.merge_aligned_parameters()
        
        # Assuming the first model as the base for merged model
        return self.models[0]


class DAREMergeMethod(nn.Module):
    """
    A class representing the DARE merge method.

    This merge method combines a base model and a fine-tuned model by adjusting the parameters
    based on their differences.

    Args:
        base_model (nn.Module): The base model.
        fine_tuned_model (nn.Module): The fine-tuned model.
g
    Attributes:
        base_model (nn.Module): The base model.
        fine_tuned_model (nn.Module): The fine-tuned model.
    """

    def __init__(
        self,
        base_model: nn.Module,
        fine_tuned_model: nn.Module,
    ):
        super(DAREMergeMethod, self).__init__()
        self.base_model = base_model
        self.fine_tuned_model = fine_tuned_model
    
    def merge(
        self,
        threshold: float = 0.01,
        amplification_factor: float = 2.0
    ):
        """
        Merge the base model and the fine-tuned model.

        This method adjusts the parameters of the fine-tuned model based on the differences
        with the base model. Parameters with small differences below the threshold are set to 0,
        while parameters with larger differences are amplified by the amplification factor.

        Args:
            threshold (float, optional): The threshold for small differences. Defaults to 0.01.
            amplification_factor (float, optional): The amplification factor for large differences. Defaults to 2.0.

        Returns:
            nn.Module: The merged fine-tuned model.
        """
        with torch.no_grad():
            for base_param, fine_tuned_param in zip(self.base_model.parameters(), self.fine_tuned_models.parameters()):
                difference = fine_tuned_param - base_param
                small_differences_mask = torch.abs(difference) < threshold
                difference[small_differences_mask] = 0.0
                difference[-small_differences_mask]  *= amplification_factor
                fine_tuned_param.copy_(base_param + difference)
        
        return self.fine_tuned_model