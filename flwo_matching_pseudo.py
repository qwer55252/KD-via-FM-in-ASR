import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingModule(nn.Module):
    def __init__(self,...):
        super().__init__()
        self.meta_encoder:nn.Module = (...)
        self.metric_based_loss_function:nn.Module = (...)
        self.time_embed:nn.Module = nn.Linear(...) 
        # 입력 : student feature와 같은 shape이고, 모든 요소가 sampling step인 tensor
        # 출력 : time embedding vector, shape: (batch_size, time_embed_dim)
        
        self.training_sampling:int = (...) # the number of sampling steps during training
        self.shape_transformation_function:nn.Module = (...)
        self.dirac_ratio:float = (...) # hyperparameter βd, which belongs to [0,1]
        self.weight:float = (...)
    
    def forward(self, s_f, t_f=None, target=None, inference_sampling=1):
        # s_f: the feature/logit of the student
        # t_f: the feature/logit of the teacher
        # target: the logit-based ground truth label, only used for logit-based distillation
        # inference_sampling: the number of sampling steps during inference
        all_p_t_f = []
        
        if self.training:
            # Shuffle one-to-one teacher-student feature/logit pair
            if t_f is not None:
                l = int(self.dirac_ratio * t_f.shape[0])
                t_f[l:][torch.randperm(t_f.shape[0] - l)] = t_f[l:].clone()
            loss, x = 0., s_f
            indices = reversed(range(1, self.training_sampling + 1))
            # Calculate the FM-KT loss
            for i in indices:
                t = torch.ones(s_f.shape[0]) * i / self.training_sampling # 이렇게 하면 cheating하는거 아니야?
                embed_t = self.time_embed(t)
                embed_x = x + embed_t
                velocity = self.meta_encoder(embed_x)
                x = x - velocity / self.training_sampling
                p_t_f = self.shape_transformation_function(s_f - velocity)
                all_p_t_f.append(p_t_f)
                loss += self.metric_based_loss_function(p_t_f, t_f)
                if target is not None:
                    loss += F.cross_entropy(p_t_f, target)
            loss *= (self.weight / self.training_sampling)
            return loss, torch.stack(all_p_t_f, 0).mean(0)
        
        else:
            x = s_f
            indices = reversed(range(1, inference_sampling + 1))
            for i in indices:
                t = torch.ones(s_f.shape[0]) * i / inference_sampling
                embed_t = self.time_embed(t)
                embed_x = x + embed_t
                velocity = self.meta_encoder(embed_x)
                x = x - velocity / inference_sampling
                all_p_t_f.append(self.shape_transformation_function(s_f - velocity))
            return 0., torch.stack(all_p_t_f, 0).mean(0)