#!/usr/bin/env python3
"""
Comprehensive Model Evaluation - Test on all videos and save to CSV.
"""

import os
import sys
import csv
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from models import ISLTranslator
from data.dataset import ISLCSLTRDataset


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    model = ISLTranslator(
        t5_model_name='t5-small',
        freeze_i3d=True,
        lstm_hidden=512,
        lstm_layers=2
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    return model


def evaluate_all_videos(model, dataset_path: str, output_csv: str, device: str = 'cuda'):
    """Evaluate model on all videos and save results to CSV."""
    
    # Load dataset (all splits)
    all_results = []
    
    for split in ['train', 'val', 'test']:
        print(f"\nEvaluating {split} split...")
        
        dataset = ISLCSLTRDataset(
            dataset_dir=dataset_path,
            split=split,
            num_frames=30,
            frame_size=224
        )
        
        for idx in tqdm(range(len(dataset)), desc=split):
            try:
                sample = dataset[idx]
                video = sample['video'].unsqueeze(0).to(device)
                reference = sample['text']
                video_path = dataset.samples[idx]['video']
                
                # Generate prediction
                with torch.no_grad():
                    prediction = model.translate(video, num_beams=4, max_length=50)[0]
                
                # Check if correct
                is_correct = prediction.lower().strip() == reference.lower().strip()
                
                all_results.append({
                    'split': split,
                    'video_path': video_path,
                    'reference': reference,
                    'prediction': prediction,
                    'is_correct': is_correct,
                    'reference_length': len(reference.split()),
                    'prediction_length': len(prediction.split()) if prediction else 0
                })
                
            except Exception as e:
                all_results.append({
                    'split': split,
                    'video_path': dataset.samples[idx]['video'],
                    'reference': dataset.samples[idx]['text'],
                    'prediction': f"ERROR: {str(e)}",
                    'is_correct': False,
                    'reference_length': 0,
                    'prediction_length': 0
                })
    
    # Save to CSV
    print(f"\nSaving results to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'split', 'video_path', 'reference', 'prediction', 
            'is_correct', 'reference_length', 'prediction_length'
        ])
        writer.writeheader()
        writer.writerows(all_results)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_results = [r for r in all_results if r['split'] == split]
        correct = sum(1 for r in split_results if r['is_correct'])
        total = len(split_results)
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"{split.upper():6} | Total: {total:4} | Correct: {correct:4} | Accuracy: {accuracy:.1f}%")
    
    # Overall
    total_correct = sum(1 for r in all_results if r['is_correct'])
    total = len(all_results)
    print("-"*60)
    print(f"{'TOTAL':6} | Total: {total:4} | Correct: {total_correct:4} | Accuracy: {total_correct/total*100:.1f}%")
    
    # Show some examples
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    for i, r in enumerate(all_results[:10]):
        print(f"\n[{i+1}] {r['split'].upper()}")
        print(f"  Ref:  {r['reference']}")
        print(f"  Pred: {r['prediction']}")
        print(f"  {'✓' if r['is_correct'] else '✗'}")
    
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/server_best_model.pt')
    parser.add_argument('--dataset', type=str, default='data/isl_clstr/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Videos_Sentence_Level')
    parser.add_argument('--output', type=str, default='evaluation_results.csv')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Evaluate
    results = evaluate_all_videos(model, args.dataset, args.output, args.device)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
