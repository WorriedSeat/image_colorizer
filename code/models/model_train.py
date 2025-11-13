import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from code.dataset.dataset import get_dataloaders
from code.models.losses import validate, lab_to_rgb_tensor, VGGPerceptual
from code.models.colorizer import ColorizerNet

def train(lr:float=1e-3, epochs:int=30, max_patience:int=10, best_model_pth:str='models/best.pt'):
    #Setting the device
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    train_loader, val_loader = get_dataloaders() # get data loaders
    
    model = ColorizerNet() #Creating a model
    optimizer = optim.Adam(model.parameters(), lr=lr) #Adam optimizer
    l1_loss = nn.L1Loss() #L1 loss
    perc_loss = VGGPerceptual(device) #VGG preceptual loss
    
    best_val_loss = 0
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        total_train_loss = 0
        
        for L, ab in loop:
            L, ab = L.to(device), ab.to(device)
            pred_ab = model(L)
            loss_l1 = l1_loss(pred_ab, ab)
            
            with torch.no_grad():
                target_rgb = lab_to_rgb_tensor(L, ab).to(device)
            pred_rgb = lab_to_rgb_tensor(L, pred_ab).to(device)
            
            loss_perc = perc_loss(pred_rgb, target_rgb)
            loss = loss_l1 + 0.1 * loss_perc
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            loop.set_postfix(train_loss=loss.item())
        
        #Calculating average train loss     
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"avg TRAIN loss for epoch {epoch+1}: {avg_train_loss:.4f}")
        
        #Calculating validation loss
        val_loss, val_l1, val_perc = validate(model, val_loader, device, l1_loss, perc_loss)
        print(f"VALIDATION loss: {val_loss:.4f} (L1: {val_l1:.4f}, Perc: {val_perc:.4f})")
        
        #Early stopping criteria
        if val_loss >= best_val_loss:
            if (val_loss - best_val_loss) > 0.005:
                patience += 1
        else:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state.dict(), best_model_pth)
        
        if patience >= max_patience:
            print(f"{'='*50}\nEarly stopping was triggered after {max_patience} epochs:\n\tBest validation loss: {val_loss:.4f} (L1: {val_l1:.4f}, Perc: {val_perc:.4f})\n\tBest model saved to {best_model_pth}!")
            break
    
    return model

if __name__ == "__main__":
    train()