#!/usr/bin/env python3
"""
Analyze All Available Patients in BreakHis Dataset

This script finds ALL patients in the dataset, including those with incomplete magnifications,
to maximize the sample size for better generalization.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

from config import FOLD_PATH
from datasets.examine import extract_tumor_type_and_patient_id

def analyze_all_patients():
    """Analyze all patients including those with incomplete magnifications"""
    
    print("🔍 ANALYZING ALL AVAILABLE PATIENTS")
    print("=" * 80)
    
    # Load the original folds CSV
    folds_df = pd.read_csv(FOLD_PATH)
    print(f"📊 Total image samples in dataset: {len(folds_df)}")
    
    # Extract patient information
    folds_df['tumor_class'], folds_df['tumor_type'], folds_df['patient_id'], folds_df['magnification'] = \
        zip(*folds_df['filename'].apply(extract_tumor_type_and_patient_id))
    
    folds_df['magnification'] = pd.to_numeric(folds_df['magnification'], errors='coerce')
    
    # Get all unique patients
    all_patients = folds_df.groupby('patient_id').agg({
        'tumor_class': 'first',
        'tumor_type': 'first',
        'magnification': lambda x: sorted(x.unique())
    }).reset_index()
    
    # Add magnification availability analysis
    all_patients['available_mags'] = all_patients['magnification'].apply(set)
    all_patients['num_mags'] = all_patients['available_mags'].apply(len)
    all_patients['has_all_4_mags'] = all_patients['num_mags'] == 4
    
    print(f"📈 Patient Statistics:")
    print(f"   Total unique patients: {len(all_patients)}")
    print(f"   Patients with all 4 magnifications: {all_patients['has_all_4_mags'].sum()}")
    print(f"   Patients with 1-3 magnifications: {len(all_patients) - all_patients['has_all_4_mags'].sum()}")
    
    # Magnification availability breakdown
    print(f"\n🔬 Magnification Availability:")
    mag_counts = all_patients['num_mags'].value_counts().sort_index()
    for num_mags, count in mag_counts.items():
        percentage = count / len(all_patients) * 100
        print(f"   {num_mags} magnifications: {count} patients ({percentage:.1f}%)")
    
    # Class distribution for all patients
    print(f"\n🏥 Class Distribution (All Patients):")
    class_dist = all_patients['tumor_class'].value_counts()
    for cls, count in class_dist.items():
        percentage = count / len(all_patients) * 100
        print(f"   {cls.capitalize()}: {count} patients ({percentage:.1f}%)")
    
    # Tumor type distribution
    print(f"\n🦠 Tumor Type Distribution:")
    tumor_dist = all_patients.groupby(['tumor_class', 'tumor_type']).size()
    for (cls, typ), count in tumor_dist.items():
        print(f"   {cls} - {typ}: {count} patients")
    
    # Show patients with incomplete magnifications
    incomplete_patients = all_patients[all_patients['num_mags'] < 4]
    if len(incomplete_patients) > 0:
        print(f"\n📉 Patients with Incomplete Magnifications ({len(incomplete_patients)} total):")
        for _, patient in incomplete_patients.iterrows():
            available_mags = sorted(list(patient['available_mags']))
            missing_mags = sorted(list({40, 100, 200, 400} - patient['available_mags']))
            print(f"   {patient['patient_id']}: has {available_mags}, missing {missing_mags}")
    
    # Calculate potential dataset increase
    current_complete = all_patients['has_all_4_mags'].sum()
    potential_total = len(all_patients)
    increase = potential_total - current_complete
    
    print(f"\n📊 Dataset Expansion Potential:")
    print(f"   Current (complete only): {current_complete} patients")
    print(f"   Potential (with fallback): {potential_total} patients")
    print(f"   Increase: +{increase} patients ({increase/current_complete*100:.1f}% more)")
    
    return all_patients, incomplete_patients

def get_fallback_magnification(available_mags, target_mag):
    """Get the closest available magnification for a target"""
    available_mags = list(available_mags)
    
    if target_mag in available_mags:
        return target_mag
    
    # Find closest by absolute difference
    closest = min(available_mags, key=lambda x: abs(x - target_mag))
    return closest

def simulate_fallback_strategy(all_patients):
    """Simulate how the fallback strategy would work"""
    
    print(f"\n🔄 SIMULATING FALLBACK MAGNIFICATION STRATEGY")
    print("=" * 80)
    
    target_mags = [40, 100, 200, 400]
    fallback_mapping = defaultdict(Counter)
    
    for _, patient in all_patients.iterrows():
        patient_id = patient['patient_id']
        available = patient['available_mags']
        
        for target in target_mags:
            fallback = get_fallback_magnification(available, target)
            if fallback != target:
                fallback_mapping[target][fallback] += 1
    
    print(f"📋 Fallback Mapping Summary:")
    for target_mag in target_mags:
        fallbacks = fallback_mapping[target_mag]
        if fallbacks:
            print(f"   {target_mag}x magnification:")
            for fallback_mag, count in fallbacks.most_common():
                print(f"     → Falls back to {fallback_mag}x for {count} patients")
        else:
            print(f"   {target_mag}x magnification: No fallbacks needed")
    
    # Calculate fallback statistics
    total_fallbacks = sum(sum(counter.values()) for counter in fallback_mapping.values())
    total_mag_requests = len(all_patients) * 4  # 4 mags per patient
    fallback_rate = total_fallbacks / total_mag_requests * 100
    
    print(f"\n📈 Fallback Statistics:")
    print(f"   Total magnification requests: {total_mag_requests}")
    print(f"   Requests requiring fallback: {total_fallbacks}")
    print(f"   Fallback rate: {fallback_rate:.1f}%")
    print(f"   Direct availability rate: {100-fallback_rate:.1f}%")
    
    return fallback_mapping

def visualize_patient_analysis(all_patients):
    """Create visualizations for patient analysis"""
    
    print(f"\n🎨 Creating visualizations...")
    os.makedirs('figs', exist_ok=True)
    
    # 1. Magnification availability
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: Number of magnifications per patient
    mag_counts = all_patients['num_mags'].value_counts().sort_index()
    axes[0, 0].bar(mag_counts.index, mag_counts.values)
    axes[0, 0].set_title('Patients by Number of Available Magnifications')
    axes[0, 0].set_xlabel('Number of Magnifications')
    axes[0, 0].set_ylabel('Number of Patients')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Class distribution
    class_counts = all_patients['tumor_class'].value_counts()
    axes[0, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Class Distribution (All Patients)')
    
    # Subplot 3: Tumor type distribution
    tumor_dist = all_patients.groupby(['tumor_class', 'tumor_type']).size().reset_index(name='count')
    benign_data = tumor_dist[tumor_dist['tumor_class'] == 'benign']
    malignant_data = tumor_dist[tumor_dist['tumor_class'] == 'malignant']
    
    x_pos_benign = np.arange(len(benign_data))
    x_pos_malignant = np.arange(len(benign_data), len(benign_data) + len(malignant_data))
    
    axes[1, 0].bar(x_pos_benign, benign_data['count'], label='Benign', alpha=0.7, color='lightblue')
    axes[1, 0].bar(x_pos_malignant, malignant_data['count'], label='Malignant', alpha=0.7, color='lightcoral')
    
    all_types = list(benign_data['tumor_type']) + list(malignant_data['tumor_type'])
    axes[1, 0].set_xticks(np.arange(len(all_types)))
    axes[1, 0].set_xticklabels(all_types, rotation=45, ha='right')
    axes[1, 0].set_title('Tumor Type Distribution')
    axes[1, 0].set_ylabel('Number of Patients')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Complete vs Incomplete magnifications by class
    completeness_by_class = all_patients.groupby(['tumor_class', 'has_all_4_mags']).size().unstack(fill_value=0)
    completeness_by_class.plot(kind='bar', ax=axes[1, 1], width=0.7)
    axes[1, 1].set_title('Magnification Completeness by Class')
    axes[1, 1].set_xlabel('Tumor Class')
    axes[1, 1].set_ylabel('Number of Patients')
    axes[1, 1].legend(['Incomplete (<4 mags)', 'Complete (4 mags)'])
    axes[1, 1].tick_params(axis='x', rotation=0)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figs/all_patients_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to figs/all_patients_analysis.png")

def main():
    """Main analysis function"""
    
    # Step 1: Analyze all patients
    all_patients, incomplete_patients = analyze_all_patients()
    
    # Step 2: Simulate fallback strategy
    fallback_mapping = simulate_fallback_strategy(all_patients)
    
    # Step 3: Create visualizations
    visualize_patient_analysis(all_patients)
    
    # Step 4: Provide recommendations
    print(f"\n" + "=" * 80)
    print(f"💡 RECOMMENDATIONS")
    print(f"=" * 80)
    
    current_patients = all_patients['has_all_4_mags'].sum()
    total_patients = len(all_patients)
    
    print(f"✅ Using fallback strategy would increase dataset by {total_patients - current_patients} patients")
    print(f"✅ This represents a {(total_patients - current_patients)/current_patients*100:.1f}% increase")
    print(f"✅ Fallback rate is low ({sum(sum(counter.values()) for counter in fallback_mapping.values())/(total_patients*4)*100:.1f}%), indicating good magnification availability")
    print(f"✅ Strategy maintains multi-magnification architecture while maximizing data")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Update MultiMagnificationDataset to support fallback strategy")
    print(f"2. Create new holdout split using all {total_patients} patients")
    print(f"3. Test with larger, more diverse patient cohort")
    print(f"4. Expect more realistic performance (likely lower than 100%)")
    
    return all_patients, incomplete_patients, fallback_mapping

if __name__ == "__main__":
    all_patients, incomplete_patients, fallback_mapping = main()