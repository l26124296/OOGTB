import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sys
import time

# [修正點 1] 改用 QApplication
from PyQt6.QtWidgets import QApplication

# 引用您的核心模組
from core.Transmission import GFCalculator, TransportAnalyzer

# ==============================================================================
# 1. Mock Data
# ==============================================================================
class MockHamiltonianModel:
    def __init__(self, N=50, t=1.0):
        self.N = N
        self.t = t
        diagonals = [np.ones(N-1) * (-t), np.ones(N-1) * (-t)]
        offsets = [1, -1]
        self.H_sparse = sp.diags(diagonals, offsets, shape=(N, N), format='csr')
        print(f"[Mock] Created 1D Chain Hamiltonian with N={N} sites.")

# ==============================================================================
# 2. 測試流程
# ==============================================================================
def run_test():
    # A. 設定參數
    N_SITES = 30
    t_hopping = 1.0
    
    mock_model = MockHamiltonianModel(N=N_SITES, t=t_hopping)
    
    lead_config = {
        'mode': '1D',
        'leads': {
            0 : 'L',
            N_SITES-1 : 'R'
        },
        'wbl_gamma': 1.0,
        'coupling': 1,
        'chain_hopping': 1.0
    }
    
    energies = np.linspace(-3.0, 3.0, 101)
    
    # --------------------------------------------------------------------------
    # B. 執行 GFCalculator
    # --------------------------------------------------------------------------
    print("\n[Step 1] Running GFCalculator...")
    t0 = time.time()
    
    # 這裡會用到 QThread，所以必須確保 QApplication 已經建立
    calculator = GFCalculator(mock_model, lead_config, energies)
    calculator.run() # 同步執行測試
    
    gf_bank = calculator.bank
    if gf_bank is None:
        print("Error: GFCalculator failed to produce a bank.")
        return

    print(f"   -> Calculation finished in {time.time() - t0:.2f} seconds.")
    print(f"   -> Storage used: {'Disk (Memmap)' if gf_bank.use_disk else 'RAM'}")

    # --------------------------------------------------------------------------
    # C. 測試 TransportAnalyzer
    # --------------------------------------------------------------------------
    analyzer = TransportAnalyzer(gf_bank, lead_config)
    
    # Test 1: 標準 Fisher-Lee
    print("\n[Step 2] Calculating Standard Transmission (L -> R)...")
    T_LR = analyzer.calculate_transmission([0], [N_SITES-1])
    
    # Test 2: 相干傳輸
    print("[Step 3] Calculating Coherent Transmission (Input: {'0: 1.0})...")
    T_Coh = analyzer.calculate_coherent_transmission({0: 1.0}, [N_SITES-1])
    
    
    # --------------------------------------------------------------------------
    # D. 視覺化
    # --------------------------------------------------------------------------
    print("\n[Step 5] Plotting results...")
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    
    # Plot 1: Transmission
    ax[0].plot(energies, T_LR, 'b-', label='Standard (Fisher-Lee)', linewidth=2)
    ax[0].plot(energies, T_Coh, 'r--', label='Coherent (Rank-1)')
    ax[0].axvline(-2*t_hopping, color='k', linestyle=':', alpha=0.5)
    ax[0].axvline(2*t_hopping, color='k', linestyle=':', alpha=0.5)
    ax[0].set_title('Transmission Probability T(E)')
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot 2: Difference
    diff = np.abs(T_LR - T_Coh)
    ax[1].plot(energies, diff, 'k-')
    ax[1].set_title('Difference (Standard vs Coherent)')
    ax[1].set_ylabel('|Delta T|')
    ax[1].grid(True)
    
    
    plt.tight_layout()
    plt.show()

    gf_bank.cleanup()

if __name__ == "__main__":
    # [修正點 2] 初始化 QApplication
    app = QApplication(sys.argv)
    
    run_test()