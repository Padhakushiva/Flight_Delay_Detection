#!/usr/bin/env python3
"""
🚀 Flight Delay Dashboard Launcher
Simple script to launch the interactive dashboard
"""

import subprocess
import sys
import os
import webbrowser
import time

def main():
    print("🛫 Flight Delay Prediction Dashboard Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("flight_delay_dashboard.py"):
        print("❌ Dashboard file not found!")
        print("💡 Please run this from the ML_Project directory")
        return
    
    # Check if models exist
    if not os.path.exists("models/best_model.pkl"):
        print("⚠️  Model files not found!")
        print("📋 Please run the notebook cells first to generate models:")
        print("   1. Execute all cells in Untitled.ipynb")
        print("   2. Make sure the model saving cell runs successfully")
        print("   3. Then run this launcher")
        return
    
    print("✅ Model files found!")
    print("🚀 Launching dashboard...")
    
    # Launch streamlit
    try:
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "flight_delay_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except FileNotFoundError:
        print("❌ Streamlit not found!")
        print("📦 Installing streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed! Please run this script again.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()