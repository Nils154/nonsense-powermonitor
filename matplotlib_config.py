#!/usr/bin/env python3
"""
Matplotlib configuration for Cursor environment
This ensures consistent plotting behavior across the power monitor project
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def configure_matplotlib():
    """
    Configure matplotlib for optimal display in Cursor environment
    """
    # Set the backend explicitly for Cursor environment
    # Qt5Agg works well with Cursor's integrated display
    matplotlib.use('Qt5Agg')
    
    # Configure matplotlib for better display
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14
    
    # Enable interactive mode for better user experience
    plt.ion()
    
    print(f"Matplotlib configured with backend: {matplotlib.get_backend()}")
    print("Interactive mode enabled")
    
    return True

def test_plotting():
    """
    Test that plotting works correctly with multiple plot types
    """
    try:
        # Create a figure with multiple subplots to demonstrate functionality
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Matplotlib Configuration Test - Multiple Plot Types', fontsize=16)
        
        # Plot 1: Simple line plot
        ax1 = axes[0, 0]
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 1, 5, 3, 6, 2, 8, 4, 7]
        ax1.plot(x, y, 'b-o', linewidth=2, markersize=6, label='Data Series')
        ax1.set_title('Line Plot')
        ax1.set_xlabel('X values')
        ax1.set_ylabel('Y values')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Scatter plot
        ax2 = axes[0, 1]
        np.random.seed(42)
        x_scatter = np.random.randn(50)
        y_scatter = 2 * x_scatter + np.random.randn(50)
        ax2.scatter(x_scatter, y_scatter, alpha=0.6, c='red', s=50)
        ax2.set_title('Scatter Plot')
        ax2.set_xlabel('X values')
        ax2.set_ylabel('Y values')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Histogram
        ax3 = axes[1, 0]
        data = np.random.normal(100, 15, 1000)
        ax3.hist(data, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax3.set_title('Histogram')
        ax3.set_xlabel('Values')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Bar chart
        ax4 = axes[1, 1]
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [23, 45, 56, 78, 32]
        bars = ax4.bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
        ax4.set_title('Bar Chart')
        ax4.set_xlabel('Categories')
        ax4.set_ylabel('Values')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the test plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"plots/matplotlib_test_{timestamp}.png"
        os.makedirs("plots", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Test plot saved to: {save_path}")
        
        # Show the plot (non-blocking for testing)
        plt.show(block=False)
        print("✓ Multi-plot test created successfully")
        print("📊 You should see a window with 4 different plot types")
        
        # Keep the plot open for a moment to see it
        plt.pause(10)
        
        # Don't close the plot - let it stay open after program ends
        # plt.close(fig)  # Commented out to keep window open
        return True
        
    except Exception as e:
        print(f"✗ Error creating test plot: {e}")
        return False

def demo_with_power_data():
    """
    Demonstrate how to use matplotlib configuration with actual power monitor data
    """
    try:
        import pandas as pd
        import glob
        
        print("\n🔧 Loading power monitor data...")
        
        # Load a sample of power data
        dataframes = []
        for file in glob.glob('./data/event*'):  
            df = pd.read_csv(file)
            dataframes.append(df)
            print(f"📁 Loaded {file}: {df.shape}")
        
        if not dataframes:
            print("❌ No power data found. ")
            # forget it
        else:
            df = pd.concat(dataframes, ignore_index=True)
            print(f"📊 Combined data shape: {df.shape}")
        
            # Create a power analysis plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Power Monitor Data Analysis Demo', fontsize=16)
            
            # Plot 1: Power timeline
            ax1 = axes[0, 0]
            if 'extreme_power' in df.columns:
                timestamps = pd.to_datetime(df['timeStamp'])
                times_float = [t.hour + t.minute/60 + t.second/3600 for t in timestamps.dt.time]
                ax1.scatter(times_float, df['extreme_power'], alpha=0.6, s=30)
                ax1.set_title('Power Events Timeline')
                ax1.set_xlabel('Time of Day')
                ax1.set_ylabel('Extreme Power')
                ax1.grid(True, alpha=0.3)
            else:
                # Fallback if no extreme_power column
                power_cols = [col for col in df.columns if col.startswith('P')][:5]
                for col in power_cols:
                    ax1.plot(df[col].head(20), label=col)
                ax1.set_title('Power Measurements (First 20 events)')
                ax1.set_xlabel('Event Index')
                ax1.set_ylabel('Power Value')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Power distribution
            ax2 = axes[0, 1]
            if 'extreme_power' in df.columns:
                ax2.hist(df['extreme_power'], bins=20, alpha=0.7, color='blue', edgecolor='black')
                ax2.set_title('Power Distribution')
                ax2.set_xlabel('Extreme Power Value')
                ax2.set_ylabel('Frequency')
            else:
                # Use first power column as fallback
                power_col = [col for col in df.columns if col.startswith('P')][0]
                ax2.hist(df[power_col], bins=20, alpha=0.7, color='blue', edgecolor='black')
                ax2.set_title(f'{power_col} Distribution')
                ax2.set_xlabel('Power Value')
                ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Power correlation
            ax3 = axes[1, 0]
            power_cols = [col for col in df.columns if col.startswith('P')][:10]  # First 10 power columns
            if len(power_cols) >= 2:
                ax3.scatter(df[power_cols[0]], df[power_cols[1]], alpha=0.6)
                ax3.set_title(f'Power Correlation: {power_cols[0]} vs {power_cols[1]}')
                ax3.set_xlabel(power_cols[0])
                ax3.set_ylabel(power_cols[1])
            else:
                ax3.text(0.5, 0.5, 'Not enough power columns\nfor correlation plot', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Power Correlation')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Event count over time
            ax4 = axes[1, 1]
            if 'timeStamp' in df.columns:
                df['date'] = pd.to_datetime(df['timeStamp']).dt.date
                daily_counts = df['date'].value_counts().sort_index()
                ax4.plot(daily_counts.index, daily_counts.values, 'o-', linewidth=2, markersize=6)
                ax4.set_title('Events per Day')
                ax4.set_xlabel('Date')
                ax4.set_ylabel('Number of Events')
                ax4.tick_params(axis='x', rotation=45)
            else:
                ax4.text(0.5, 0.5, 'No timestamp data\navailable', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Events per Day')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the demo plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"plots/power_analysis_demo_{timestamp}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Power analysis demo saved to: {save_path}")
            
            # Show the plot
            plt.show(block=False)
            print("📊 Power analysis demo plot should be visible")
            
            # Keep the plot open for a moment
            plt.pause(60)
            plt.close(fig)
            
            return True
        
    except Exception as e:
        print(f"✗ Error creating power data demo: {e}")
        return False

def demo_persistent_plots():
    """
    Demonstrate different ways to keep plot windows open after program ends
    """
    print("\n🔧 Demonstrating persistent plot windows...")
    
    # Method 1: Simple approach - don't close figures
    print("\n📊 Method 1: Keep plots open by not closing them")
    try:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 1, 5, 3]
        ax1.plot(x, y, 'b-o', linewidth=2, markersize=8)
        ax1.set_title('Persistent Plot - Method 1 (No close)')
        ax1.set_xlabel('X values')
        ax1.set_ylabel('Y values')
        ax1.grid(True, alpha=0.3)
        
        plt.show(block=False)
        print("✓ Plot 1 created - will stay open after program ends")
        
        # Don't close this figure - it will persist
        # plt.close(fig1)  # Commented out
        
    except Exception as e:
        print(f"✗ Method 1 failed: {e}")
    
    # Method 2: Use plt.ion() for interactive mode
    print("\n📊 Method 2: Interactive mode with plt.ion()")
    try:
        plt.ion()  # Turn on interactive mode
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        y = [3, 1, 4, 1, 5, 9, 2, 6]
        ax2.plot(x, y, 'r-s', linewidth=2, markersize=8)
        ax2.set_title('Persistent Plot - Method 2 (Interactive)')
        ax2.set_xlabel('X values')
        ax2.set_ylabel('Y values')
        ax2.grid(True, alpha=0.3)
        
        plt.show(block=False)
        print("✓ Plot 2 created in interactive mode - will stay open")
        
        # Interactive plots persist automatically
        # plt.close(fig2)  # Commented out
        
    except Exception as e:
        print(f"✗ Method 2 failed: {e}")
    
    # Method 3: Use blocking show() at the end
    print("\n📊 Method 3: Blocking show() - keeps program running")
    try:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        x = [1, 2, 3, 4, 5, 6]
        y = [2, 7, 1, 8, 2, 8]
        ax3.plot(x, y, 'g-^', linewidth=2, markersize=8)
        ax3.set_title('Persistent Plot - Method 3 (Blocking)')
        ax3.set_xlabel('X values')
        ax3.set_ylabel('Y values')
        ax3.grid(True, alpha=0.3)
        
        # This will keep the program running until window is closed
        print("✓ Plot 3 created - program will wait for you to close the window")
        print("   (Close the window to continue, or press Ctrl+C to exit)")
        
        # Uncomment the line below to use blocking mode
        # plt.show(block=True)  # This keeps program running
        plt.show(block=False)  # For demo, we'll use non-blocking
        
    except Exception as e:
        print(f"✗ Method 3 failed: {e}")
    
    print("\n✅ Persistent plot demonstration complete!")
    print("📊 Plots should remain open after this program ends")
    print("💡 Tips:")
    print("   - Method 1: Simply don't call plt.close()")
    print("   - Method 2: Use plt.ion() for interactive mode")
    print("   - Method 3: Use plt.show(block=True) to keep program running")
    print("   - All methods save plots to files as backup")

def keep_plots_open():
    """
    Utility function to keep all plots open after program ends
    Call this at the end of your analysis scripts
    """
    print("\n🔧 Configuring plots to stay open...")
    
    # Enable interactive mode
    plt.ion()
    
    # Get all existing figures
    figures = [plt.figure(i) for i in plt.get_fignums()]
    
    if figures:
        print(f"📊 Found {len(figures)} existing plots")
        print("📊 Plots will remain open after program ends")
        print("💡 Close plot windows manually when done, or press Ctrl+C")
        
        # Show all figures
        for i, fig in enumerate(figures):
            fig.show()
            print(f"   - Figure {i+1}: {fig.get_label() or 'Untitled'}")
    else:
        print("📊 No plots found to keep open")
    
    return figures

if __name__ == "__main__":
    print("Configuring matplotlib for Cursor environment...")
    configure_matplotlib()

    demo_with_power_data()
    
    print("\n" + "="*50)
    print("DEMO: Persistent Plot Windows")
    print("="*50)
    demo_persistent_plots()
    
    print("\n✅ Matplotlib configuration and demo complete!")
    print("📁 Check the 'plots' directory for saved images.")
    print("📊 Plot windows should be visible (if display is available).")
