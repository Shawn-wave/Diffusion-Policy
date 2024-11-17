import torch
import h5py
import wandb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from diffusers.optimization import get_scheduler
from tqdm import tqdm

from lerobot.config.configuration_diffusion import DiffusionConfig
from lerobot.model.modeling_diffusion import DiffusionPolicy

class RobotDataset(Dataset):
    def __init__(self, h5_path, config: DiffusionConfig):
        super().__init__()
        self.config = config
        
        # h5 파일에서 데이터 로드
        with h5py.File(h5_path, 'r') as f:
            self.images = np.array(f['images'])  # [N, H, W, C]
            self.states = np.array(f['states'])  # [N, state_dim]
            self.actions = np.array(f['actions'])  # [N, 6] (x,y,z,roll,pitch,yaw)
        
        # 데이터 통계 계산 (정규화용)
        self.dataset_stats = {
            'observation.image': {
                'mean': np.mean(self.images, axis=(0,1,2)),
                'std': np.std(self.images, axis=(0,1,2))
            },
            'observation.state': {
                'min': np.min(self.states, axis=0),
                'max': np.max(self.states, axis=0)
            },
            'action': {
                'min': np.min(self.actions, axis=0),
                'max': np.max(self.actions, axis=0)
            }
        }
        
        # 유효한 시퀀스 인덱스 계산
        self.valid_indices = []
        total_steps = len(self.images)
        for i in range(total_steps - (config.n_obs_steps + config.horizon) + 1):
            self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        seq_idx = self.valid_indices[idx]
        
        # 관찰 데이터 준비
        obs_images = self.images[seq_idx:seq_idx + self.config.n_obs_steps]
        obs_states = self.states[seq_idx:seq_idx + self.config.n_obs_steps]
        
        # 행동 시퀀스 준비
        actions = self.actions[seq_idx + self.config.n_obs_steps - 1:
                             seq_idx + self.config.n_obs_steps + self.config.horizon - 1]
        
        return {
            'observation.image': torch.FloatTensor(obs_images).permute(0, 3, 1, 2),
            'observation.state': torch.FloatTensor(obs_states),
            'action': torch.FloatTensor(actions),
            'action_is_pad': torch.zeros(self.config.horizon, dtype=torch.bool)
        }

def compute_pose_metrics(pred_actions, true_actions):
    """6D 포즈에 대한 메트릭 계산"""
    metrics = {
        'position': {
            'x': torch.mean(torch.abs(pred_actions[:, 0] - true_actions[:, 0])),
            'y': torch.mean(torch.abs(pred_actions[:, 1] - true_actions[:, 1])),
            'z': torch.mean(torch.abs(pred_actions[:, 2] - true_actions[:, 2]))
        },
        'rotation': {
            'roll': torch.mean(torch.abs(pred_actions[:, 3] - true_actions[:, 3])),
            'pitch': torch.mean(torch.abs(pred_actions[:, 4] - true_actions[:, 4])),
            'yaw': torch.mean(torch.abs(pred_actions[:, 5] - true_actions[:, 5]))
        }
    }
    return metrics

def train_diffusion_policy(
    model,
    train_dataset,
    val_dataset=None,
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-4,
    weight_decay=0.0,
    num_warmup_steps=500,
    save_path="diffusion_policy.pt",
    device="cuda",
    use_wandb=True
):
    """
    Enhanced training function for diffusion policy
    """
    if use_wandb:
        wandb.init(
            project="robot_diffusion_policy",
            config={
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_warmup_steps": num_warmup_steps
            }
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.95, 0.999),
        eps=1e-8
    )
    
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_metrics = {
            'loss': [], 'x_error': [], 'y_error': [], 'z_error': [],
            'roll_error': [], 'pitch_error': [], 'yaw_error': []
        }
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            loss = outputs["loss"]
            
            # Compute detailed metrics
            pred_actions = model.unnormalize_outputs({"action": outputs["pred_actions"]})["action"]
            metrics = compute_pose_metrics(pred_actions, batch['action'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Record metrics
            train_metrics['loss'].append(loss.item())
            train_metrics['x_error'].append(metrics['position']['x'].item())
            train_metrics['y_error'].append(metrics['position']['y'].item())
            train_metrics['z_error'].append(metrics['position']['z'].item())
            train_metrics['roll_error'].append(metrics['rotation']['roll'].item())
            train_metrics['pitch_error'].append(metrics['rotation']['pitch'].item())
            train_metrics['yaw_error'].append(metrics['rotation']['yaw'].item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': np.mean(train_metrics['loss'][-100:]),
                'pos_err': np.mean([
                    train_metrics['x_error'][-100:],
                    train_metrics['y_error'][-100:],
                    train_metrics['z_error'][-100:]
                ])
            })
        
        # Validation
        if val_dataset is not None:
            model.eval()
            val_metrics = {
                'loss': [], 'x_error': [], 'y_error': [], 'z_error': [],
                'roll_error': [], 'pitch_error': [], 'yaw_error': []
            }
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(batch)
                    pred_actions = model.unnormalize_outputs({"action": outputs["pred_actions"]})["action"]
                    metrics = compute_pose_metrics(pred_actions, batch['action'])
                    
                    val_metrics['loss'].append(outputs["loss"].item())
                    val_metrics['x_error'].append(metrics['position']['x'].item())
                    val_metrics['y_error'].append(metrics['position']['y'].item())
                    val_metrics['z_error'].append(metrics['position']['z'].item())
                    val_metrics['roll_error'].append(metrics['rotation']['roll'].item())
                    val_metrics['pitch_error'].append(metrics['rotation']['pitch'].item())
                    val_metrics['yaw_error'].append(metrics['rotation']['yaw'].item())
            
            val_loss = np.mean(val_metrics['loss'])
            
            # Log metrics
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'train/loss': np.mean(train_metrics['loss']),
                    'train/position/x_error': np.mean(train_metrics['x_error']),
                    'train/position/y_error': np.mean(train_metrics['y_error']),
                    'train/position/z_error': np.mean(train_metrics['z_error']),
                    'train/rotation/roll_error': np.mean(train_metrics['roll_error']),
                    'train/rotation/pitch_error': np.mean(train_metrics['pitch_error']),
                    'train/rotation/yaw_error': np.mean(train_metrics['yaw_error']),
                    'val/loss': val_loss,
                    'val/position/x_error': np.mean(val_metrics['x_error']),
                    'val/position/y_error': np.mean(val_metrics['y_error']),
                    'val/position/z_error': np.mean(val_metrics['z_error']),
                    'val/rotation/roll_error': np.mean(val_metrics['roll_error']),
                    'val/rotation/pitch_error': np.mean(val_metrics['pitch_error']),
                    'val/rotation/yaw_error': np.mean(val_metrics['yaw_error'])
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'val_loss': val_loss,
                    'dataset_stats': train_dataset.dataset_stats
                }, save_path)
                
            print(f"Epoch {epoch+1}: Train Loss = {np.mean(train_metrics['loss']):.4f}, "
                  f"Val Loss = {val_loss:.4f}")
        else:
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'train/loss': np.mean(train_metrics['loss']),
                    'train/position/x_error': np.mean(train_metrics['x_error']),
                    'train/position/y_error': np.mean(train_metrics['y_error']),
                    'train/position/z_error': np.mean(train_metrics['z_error']),
                    'train/rotation/roll_error': np.mean(train_metrics['roll_error']),
                    'train/rotation/pitch_error': np.mean(train_metrics['pitch_error']),
                    'train/rotation/yaw_error': np.mean(train_metrics['yaw_error'])
                })
            
            print(f"Epoch {epoch+1}: Train Loss = {np.mean(train_metrics['loss']):.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'train_loss': np.mean(train_metrics['loss']),
                    'dataset_stats': train_dataset.dataset_stats
                }, f"{save_path}_epoch_{epoch+1}.pt")
    
    return model

if __name__ == "__main__":
    # 설정
    config = DiffusionConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        input_shapes={
            "observation.image": [3, 240, 320],
            "observation.state": [6]
        },
        output_shapes={
            "action": [6]  # x,y,z,roll,pitch,yaw
        }
    )
    
    # 모델 및 데이터셋 생성
    model = DiffusionPolicy(config)
    train_dataset = RobotDataset('path/to/train.h5', config)
    val_dataset = RobotDataset('path/to/val.h5', config)
    
    # 학습 실행
    trained_model = train_diffusion_policy(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=32,
        num_epochs=600,
        learning_rate=1e-4,
        weight_decay=1e-6,
        num_warmup_steps=500,
        save_path="diffusion_policy_best.pt"
    )