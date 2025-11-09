import torch
import argparse
import pandas as pd
from tqdm import tqdm
import os
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
from ppnet.preprocess import mean, std

#top-k values (0-max prototypes)
K_VALUES = [0, 1, 5, 10]

def get_test_loader(dataset_path, batch_size):
    img_size = 224
    num_workers = 0
    test_dir = os.path.join(dataset_path, 'test')
    normalize = T.Normalize(mean=mean, std=std)
    test_dataset = datasets.ImageFolder(
        test_dir,
        T.Compose([
            T.Resize(size=(img_size, img_size)),
            T.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False)
    return test_loader


def run_probe(model, dataloader, device, mask_topk=0):
    """
    Runs a single probe intervention on the test set.
    
    Args:
        model: The loaded ProtoPNet model.
        dataloader: The test data loader.
        device: The device (cpu or cuda) to run on.
        mask_topk (int): The number of top-k prototypes to mask. 
                         If 0, runs the baseline (no mask).
    
    Returns:
        float: The top-1 accuracy for this intervention.
    """
    model.eval()
    correct = 0
    total = 0
    
    bar = tqdm(dataloader, desc=f"Probe (mask_topk={mask_topk})")
    
    for (images, labels) in bar:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            # 1. Get the original prototype activations (similarity scores)
            _, min_distances = model(images)
            proto_activations = model.distance_2_similarity(min_distances)
            
            # 2. Apply the mask_topk intervention
            if mask_topk > 0:
                # Find the top k activations for each image in the batch
                _, topk_indices = torch.topk(proto_activations, k=mask_topk, dim=1)
                
                # Create a mask.
                # Set the top-k indices to 0.0 (masking them out)
                # scatter_ sets values in 'mask' to 0.0 at the 'topk_indices'
                mask = torch.ones_like(proto_activations, device=device)
                mask.scatter_(dim=1, index=topk_indices, value=0.0)
                
                # Apply the mask to the activations
                masked_activations = proto_activations * mask
            else:
                # Baseline case (k=0), use original activations
                masked_activations = proto_activations
            
            # 3. Get final logits from the masked activations
            logits = model.last_layer(masked_activations)
            
            # 4. Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            bar.set_postfix({'acc': f'{100 * correct / total:.2f}%'})

    final_acc = correct / total
    return final_acc

def main():
    """
    Main function to run the mask_topk sweep.
    """
    parser = argparse.ArgumentParser(
        description="Concise sweep for 'mask_topk' intervention."
    )
    parser.add_argument(
        '--model', 
        required=True, 
        help='Path to trained ProtoPNet .pth model'
    )
    parser.add_argument(
        '--dataset', 
        required=True, 
        help='Path to root CUB-200 dataset directory'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64,
        help='Batch size for evaluation'
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    summary_data = []
    
    print(f"Loading model from {args.model}")
    model = torch.load(args.model, map_location=device)

    print(f"Loading data from {args.dataset}")
    test_loader = get_test_loader(
        args.dataset,
        batch_size=args.batch_size
    )

    for k in K_VALUES:
        print(f"\n--- Running sweep for mask_topk = {k} ---")
        accuracy = run_probe(model, test_loader, device, mask_topk=k)
        print(f"Resulting Accuracy: {accuracy:.4f}")
        
        summary_data.append({
            'mask_topk': k,
            'accuracy_top1': accuracy
        })

    summary_df = pd.DataFrame(summary_data)
    output_file = 'mask_topk_summary.csv'
    summary_df.to_csv(output_file, index=False)
    
    print(f"\nSweep complete. Results saved to {output_file}")
    print(summary_df)

if __name__ == "__main__":
    main()