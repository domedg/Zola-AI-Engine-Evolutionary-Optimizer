import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

def run_step(command, description):
    print(f"\n" + "="*60)
    print(f"🚀 STEP: {description}")
    print(f"Executing: {command}")
    print("="*60 + "\n")
    
    process = subprocess.Popen(command, shell=True, cwd=PROJECT_ROOT)
    process.wait()
    
    if process.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with exit code {process.returncode}")
        sys.exit(1)
    print(f"\n✅ SUCCESS: {description} completed.")

def main():
    print("""
    🏆 ZOLA CHAMPION TRAINING ORCHESTRATOR
    ======================================
    This script will guide you through the full training process:
    1. Position Collection (~30-60 min)
    2. Texel Tuning (~15 min)
    3. GA Refinement (Overnight recommended)
    4. Validation Tournament (20 games @ 3s)
    """)
    
    confirm = input("Do you want to start the full training? (y/n): ")
    if confirm.lower() != 'y':
        print("Training cancelled.")
        return

    # Step 1: Collection
    print("\nStarting Phase 1: Collecting training data from games...")
    run_step("python3 scripts/collect_positions.py", "Position Collection")
    
    # Step 2: Texel Tuning
    print("\nStarting Phase 2: Optimizing weights via Texel Tuning...")
    run_step("python3 scripts/texel_tune.py", "Texel Tuning")
    
    # Step 3: GA Training
    print("\nStarting Phase 3: Refining weights via Genetic Algorithm...")
    print("NOTE: This step is designed to run for a long time. You can stop it (Ctrl+C) anytime.")
    print("The best weights are saved automatically after every generation.")
    
    try:
        run_step("python3 scripts/train_champion_ga.py", "GA Refinement")
    except KeyboardInterrupt:
        print("\n\n🛑 GA Refinement stopped by user. Best weights are preserved in 'logs/best_weights_champion.json'.")

    # Step 4: Validation
    print("\nStarting Phase 4: Running validation tournament...")
    run_step("python3 scripts/validate_champion.py", "Validation Tournament")

    print("\n" + "="*60)
    print("🎉 TRAINING PROCESS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()

