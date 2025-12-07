import os
import matplotlib.pyplot as plt


class ResultsComparator:
    """Compare baseline vs augmented model performance"""
    
    def compare_accuracies(self, baseline_acc, augmented_acc, dataset_name, model_type):
        """
        Display comparison between baseline and augmented models
        
        Args:
            baseline_acc: Accuracy before augmentation
            augmented_acc: Accuracy after augmentation
            dataset_name: Dataset name
            model_type: Model architecture name
        """
        print("\n" + "=" * 70)
        print(" DATA-CENTRIC IMPROVEMENT ANALYSIS")
        print("=" * 70)
        
        improvement = augmented_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc) * 100
        
        # Calculate error reduction
        baseline_error = 1 - baseline_acc
        augmented_error = 1 - augmented_acc
        error_reduction = baseline_error - augmented_error
        
        if baseline_error > 0:
            relative_error_reduction = (error_reduction / baseline_error) * 100
        else:
            relative_error_reduction = 0
        
        print(f"\n[TEST] Model: {model_type}")
        print(f"[DATA] Dataset: {dataset_name.upper()}")
        print(f"\n{'Metric':<30} {'Baseline':<15} {'Augmented':<15} {'Change':<15}")
        print("-" * 70)
        print(f"{'Accuracy':<30} {baseline_acc:<15.4f} {augmented_acc:<15.4f} {improvement:+.4f}")
        print(f"{'Accuracy %':<30} {baseline_acc*100:<15.2f} {augmented_acc*100:<15.2f} {improvement*100:+.2f}")
        print(f"{'Error Rate':<30} {baseline_error:<15.4f} {augmented_error:<15.4f} {-error_reduction:+.4f}")
        print(f"{'Error Rate %':<30} {baseline_error*100:<15.2f} {augmented_error*100:<15.2f} {-error_reduction*100:+.2f}")
        print("-" * 70)
        print(f"{'Improvement %':<30} {'-':<15} {'-':<15} {improvement_pct:+.2f}%")
        print(f"{'Relative Error Reduction':<30} {'-':<15} {'-':<15} {relative_error_reduction:+.1f}%")
        
        if improvement > 0:
            print(f"\n[OK] SUCCESS: Data-centric approach improved accuracy!")
            print(f"   Absolute improvement: {improvement:.4f} ({improvement*100:.2f}%)")
            print(f"   Error reduction: {baseline_error*100:.2f}% → {augmented_error*100:.2f}%")
            print(f"   Relative error reduced by: {relative_error_reduction:.1f}%")
        elif improvement == 0:
            print(f"\n[WARN]  No change detected (accuracy unchanged)")
            print(f"   Possible reasons: Dataset already optimal, need more augmentation")
        else:
            print(f"\n[WARN]  Accuracy decreased by {abs(improvement):.4f}")
            print(f"   Possible causes: Overfitting, insufficient epochs, poor augmentation")
            print(f"   Recommendation: Adjust augmentation parameters or increase epochs")
        
        # Create visualization
        self._plot_comparison(baseline_acc, augmented_acc, dataset_name, model_type, improvement)
        
        return improvement
    
    def _plot_comparison(self, baseline_acc, augmented_acc, dataset_name, model_type, improvement):
        """Plot side-by-side accuracy comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Accuracy comparison
        categories = ['Baseline\n(Original Data)', 'Augmented\n(Error-Enhanced)']
        accuracies = [baseline_acc * 100, augmented_acc * 100]
        colors = ['#3498db', '#2ecc71' if improvement >= 0 else '#e74c3c']
        
        bars = ax1.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{acc:.2f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Add improvement arrow if positive
        if improvement > 0:
            ax1.annotate('', xy=(1, augmented_acc * 100), xytext=(0, baseline_acc * 100),
                       arrowprops=dict(arrowstyle='->', color='green', lw=3))
            mid_y = (baseline_acc * 100 + augmented_acc * 100) / 2
            ax1.text(0.5, mid_y, f'+{improvement*100:.2f}%', 
                   ha='center', fontsize=12, color='green', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax1.set_title(f'Accuracy Comparison\n{model_type} on {dataset_name.upper()}',
                    fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylim([min(accuracies) - 2, 100])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
        
        # Plot 2: Error rate comparison
        error_rates = [(1 - baseline_acc) * 100, (1 - augmented_acc) * 100]
        bars2 = ax2.bar(categories, error_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, err in zip(bars2, error_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{err:.2f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        ax2.set_ylabel('Error Rate (%)', fontsize=13, fontweight='bold')
        ax2.set_title(f'Error Rate Comparison\n{model_type} on {dataset_name.upper()}',
                    fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylim([0, max(error_rates) * 1.3])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/comparison_{model_type}_{dataset_name}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n[DATA] Comparison plot saved: {plot_path}")
        
        plt.show()


