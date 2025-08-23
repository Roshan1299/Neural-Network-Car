# Mutation Impact Analysis for NeuroNet_Car - REAL DATA VERSION
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# <data_loading>
# Load real mutation data from NeuroNet_Car
if not os.path.exists('mutation_data.json'):
    print("❌ No mutation_data.json found!")
    print("Run the simulation and press 'R' a few times to generate data.")
    exit(1)

with open('mutation_data.json', 'r') as f:
    real_data = json.load(f)

if len(real_data) < 3:
    print(f"❌ Need at least 3 generations for analysis. Current: {len(real_data)}")
    print("Run more generations in the simulation.")
    exit(1)

df = pd.DataFrame(real_data)
print(f"✅ Loaded {len(df)} generations of REAL data!")
print(f"Generation range: {df['generation'].min()} to {df['generation'].max()}")
print(f"Mutation rates used: {df['mutation_rate'].min():.3f} to {df['mutation_rate'].max():.3f}")
# </data_loading>

# <mutation_analysis>
# Create comprehensive mutation impact visualization
fig = plt.figure(figsize=(20, 16))

# 1. Core Scatter Plot - Mutation Rate vs Performance Improvement
ax1 = plt.subplot(3, 3, 1)
scatter = ax1.scatter(df['mutation_rate'], df['performance_improvement'], 
                     s=df['population_diversity']*3 + 50,  # Ensure visible points
                     c=df['population_diversity'], 
                     cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
ax1.set_xlabel('Mutation Rate', fontsize=12)
ax1.set_ylabel('Performance Improvement', fontsize=12)
ax1.set_title('REAL DATA: Mutation Rate vs Performance Improvement\n(Point size/color = Population Diversity)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Population Diversity')

# Add trend line if we have enough data points
if len(df) >= 3:
    z = np.polyfit(df['mutation_rate'], df['performance_improvement'], 1)  # Linear for small datasets
    p = np.poly1d(z)
    x_trend = np.linspace(df['mutation_rate'].min(), df['mutation_rate'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
    ax1.legend()

# 2. Performance Evolution Over Generations
ax2 = plt.subplot(3, 3, 2)
ax2.plot(df['generation'], df['best_fitness'], 'b-', linewidth=3, marker='o', markersize=6, label='Best Fitness')
ax2.plot(df['generation'], df['avg_fitness'], 'g--', linewidth=2, marker='s', markersize=4, label='Average Fitness')
ax2.set_xlabel('Generation', fontsize=12)
ax2.set_ylabel('Fitness Score', fontsize=12)
ax2.set_title('REAL Evolution: Performance Over Time', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Mutation Rate Changes Over Time
ax3 = plt.subplot(3, 3, 3)
ax3.plot(df['generation'], df['mutation_rate'], 'r-', linewidth=2, marker='D', markersize=5)
ax3.set_xlabel('Generation', fontsize=12)
ax3.set_ylabel('Mutation Rate', fontsize=12)
ax3.set_title('Mutation Rate Strategy Over Time', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add horizontal lines for reference
for rate, label in [(0.05, 'Low'), (0.1, 'Medium'), (0.2, 'High')]:
    if rate in df['mutation_rate'].values:
        ax3.axhline(y=rate, color='gray', linestyle=':', alpha=0.5, label=f'{label} ({rate})')
ax3.legend()

# 4. Population Success Analysis
ax4 = plt.subplot(3, 3, 4)
success_rate = (df['cars_beat_previous'] / 10) * 100  # Assuming 10 cars per generation
bars = ax4.bar(range(len(df)), success_rate, color=['green' if x > 50 else 'orange' if x > 20 else 'red' for x in success_rate])
ax4.set_xlabel('Generation Index', fontsize=12)
ax4.set_ylabel('Cars Beating Previous Gen (%)', fontsize=12)
ax4.set_title('Population Success Rate', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add average line
avg_success = success_rate.mean()
ax4.axhline(y=avg_success, color='blue', linestyle='--', linewidth=2, label=f'Average: {avg_success:.1f}%')
ax4.legend()

# 5. Mutation Rate Effectiveness
ax5 = plt.subplot(3, 3, 5)
unique_rates = df['mutation_rate'].unique()
if len(unique_rates) > 1:
    rate_performance = []
    rate_labels = []
    rate_errors = []
    
    for rate in sorted(unique_rates):
        mask = df['mutation_rate'] == rate
        improvements = df[mask]['performance_improvement']
        rate_performance.append(improvements.mean())
        rate_errors.append(improvements.std())
        rate_labels.append(f'{rate:.3f}\n({mask.sum()} gens)')
    
    bars = ax5.bar(rate_labels, rate_performance, yerr=rate_errors, 
                   capsize=10, alpha=0.7, color=['green', 'orange', 'red'][:len(rate_labels)])
    ax5.set_ylabel('Average Performance Improvement', fontsize=12)
    ax5.set_xlabel('Mutation Rate\n(Generations)', fontsize=12)
    ax5.set_title('Mutation Rate Effectiveness', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Highlight best rate
    best_idx = np.argmax(rate_performance)
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(2)
else:
    ax5.text(0.5, 0.5, f'Only one mutation rate used:\n{unique_rates[0]:.3f}', 
             ha='center', va='center', transform=ax5.transAxes, fontsize=14)
    ax5.set_title('Single Mutation Rate Used', fontsize=14, fontweight='bold')

# 6. Fitness Distribution
ax6 = plt.subplot(3, 3, 6)
ax6.hist(df['best_fitness'], bins=min(10, len(df)//2 + 1), alpha=0.7, color='blue', edgecolor='black')
ax6.set_xlabel('Best Fitness Score', fontsize=12)
ax6.set_ylabel('Frequency', fontsize=12)
ax6.set_title('Distribution of Best Fitness Scores', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Add statistics
mean_fitness = df['best_fitness'].mean()
std_fitness = df['best_fitness'].std()
ax6.axvline(mean_fitness, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_fitness:.1f}')
ax6.axvline(mean_fitness + std_fitness, color='orange', linestyle=':', linewidth=1, label=f'+1σ: {mean_fitness + std_fitness:.1f}')
ax6.axvline(mean_fitness - std_fitness, color='orange', linestyle=':', linewidth=1, label=f'-1σ: {mean_fitness - std_fitness:.1f}')
ax6.legend()

# 7. Performance Improvement Trends
ax7 = plt.subplot(3, 3, 7)
cumulative_improvement = df['performance_improvement'].cumsum()
ax7.plot(df['generation'], cumulative_improvement, 'purple', linewidth=3, marker='o', markersize=4)
ax7.set_xlabel('Generation', fontsize=12)
ax7.set_ylabel('Cumulative Improvement', fontsize=12)
ax7.set_title('Cumulative Performance Gains', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3)

# Add trend line
if len(df) >= 3:
    z = np.polyfit(df['generation'], cumulative_improvement, 1)
    p = np.poly1d(z)
    ax7.plot(df['generation'], p(df['generation']), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.1f}/gen')
    ax7.legend()

# 8. Population Diversity Analysis
ax8 = plt.subplot(3, 3, 8)
ax8.scatter(df['mutation_rate'], df['population_diversity'], 
           s=80, c=df['generation'], cmap='plasma', alpha=0.7, edgecolors='black', linewidth=0.5)
ax8.set_xlabel('Mutation Rate', fontsize=12)
ax8.set_ylabel('Population Diversity (Std Dev)', fontsize=12)
ax8.set_title('Diversity vs Mutation Rate\n(Color = Generation)', fontsize=14, fontweight='bold')
ax8.grid(True, alpha=0.3)

# Add colorbar
scatter_cb = plt.colorbar(ax8.collections[0], ax=ax8, label='Generation')

# 9. Summary Statistics Table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

# Calculate key statistics
best_generation = df.loc[df['best_fitness'].idxmax()]
worst_generation = df.loc[df['best_fitness'].idxmin()]
most_improved = df.loc[df['performance_improvement'].idxmax()]

stats_data = {
    'Metric': [
        'Total Generations',
        'Best Performance', 
        'Worst Performance',
        'Average Improvement',
        'Biggest Jump',
        'Most Used Mut Rate',
        'Best Mut Rate',
        'Current Trend'
    ],
    'Value': [
        f'{len(df)}',
        f'{best_generation["best_fitness"]:.1f} (Gen {best_generation["generation"]})',
        f'{worst_generation["best_fitness"]:.1f} (Gen {worst_generation["generation"]})',
        f'{df["performance_improvement"].mean():.1f} ± {df["performance_improvement"].std():.1f}',
        f'{most_improved["performance_improvement"]:.1f} (Gen {most_improved["generation"]})',
        f'{df["mutation_rate"].mode()[0]:.3f}',
        f'{df.loc[df["performance_improvement"].idxmax(), "mutation_rate"]:.3f}',
        '↗️ Improving' if df['best_fitness'].iloc[-1] > df['best_fitness'].iloc[0] else '↘️ Declining'
    ]
}

# Create table
table_data = []
for metric, value in zip(stats_data['Metric'], stats_data['Value']):
    table_data.append([metric, str(value)[:25]])  # Truncate long values

table = ax9.table(cellText=table_data, 
                 colLabels=['Real Data Statistics', 'Value'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

# Style the table
for i in range(len(table_data) + 1):
    for j in range(2):
        cell = table[(i, j)]
        if i == 0:  # Header row
            cell.set_facecolor('#2E8B57')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

ax9.set_title('REAL Training Statistics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout(pad=3.0)
plt.savefig('mutation_impact_analysis_REAL.png', dpi=300, bbox_inches='tight')
plt.show()
# </mutation_analysis>

# <recommendations>
print("\n" + "="*80)
print("🚗 NEURONET_CAR MUTATION ANALYSIS - REAL DATA INSIGHTS")
print("="*80)

best_rate_idx = df['performance_improvement'].idxmax()
best_rate = df.loc[best_rate_idx, 'mutation_rate']
best_improvement = df.loc[best_rate_idx, 'performance_improvement']

print(f"🎯 BEST PERFORMING MUTATION RATE: {best_rate:.3f}")
print(f"   - Achieved improvement of: {best_improvement:.1f}")
print(f"   - Occurred in generation: {df.loc[best_rate_idx, 'generation']}")

# Analyze trends
recent_data = df.tail(5) if len(df) >= 5 else df
recent_trend = recent_data['best_fitness'].iloc[-1] - recent_data['best_fitness'].iloc[0]

if recent_trend > 0:
    print(f"\n📈 RECENT TREND: POSITIVE (+{recent_trend:.1f} in last {len(recent_data)} generations)")
    print("   - Your AI is still improving!")
else:
    print(f"\n📉 RECENT TREND: PLATEAU/DECLINE ({recent_trend:.1f} in last {len(recent_data)} generations)")
    print("   - Consider trying different mutation rates or strategies")

# Rate recommendations
unique_rates = df['mutation_rate'].unique()
if len(unique_rates) > 1:
    rate_performance = {}
    for rate in unique_rates:
        mask = df['mutation_rate'] == rate
        avg_improvement = df[mask]['performance_improvement'].mean()
        rate_performance[rate] = avg_improvement
    
    best_tested_rate = max(rate_performance.keys(), key=lambda x: rate_performance[x])
    print(f"\n🏆 BEST TESTED RATE: {best_tested_rate:.3f}")
    print(f"   - Average improvement: {rate_performance[best_tested_rate]:.1f}")
    
    # Recommendations for next rates to try
    if best_tested_rate < 0.08:
        print("   💡 SUGGESTION: Try slightly higher rates (0.1-0.15) for more exploration")
    elif best_tested_rate > 0.15:
        print("   💡 SUGGESTION: Try lower rates (0.05-0.1) for more stable improvements")
    else:
        print("   💡 SUGGESTION: You're in a good range! Try fine-tuning around this rate")
else:
    print(f"\n⚠️  SINGLE MUTATION RATE USED: {unique_rates[0]:.3f}")
    print("   💡 RECOMMENDATION: Try different rates (press 1/2/3 keys) to find optimal settings")

# Performance analysis
total_improvement = df['best_fitness'].iloc[-1] - df['best_fitness'].iloc[0]
generations_span = df['generation'].iloc[-1] - df['generation'].iloc[0]
avg_improvement_per_gen = total_improvement / generations_span if generations_span > 0 else 0

print(f"\n📊 OVERALL PERFORMANCE:")
print(f"   - Total improvement: {total_improvement:.1f} over {generations_span} generations") 
print(f"   - Average per generation: {avg_improvement_per_gen:.2f}")
print(f"   - Best ever fitness: {df['best_fitness'].max():.1f}")
print(f"   - Current fitness: {df['best_fitness'].iloc[-1]:.1f}")

# Population insights
avg_diversity = df['population_diversity'].mean()
avg_survivors = df['cars_survived'].mean()

print(f"\n🚗 POPULATION INSIGHTS:")
print(f"   - Average population diversity: {avg_diversity:.1f}")
print(f"   - Average cars surviving: {avg_survivors:.1f}/10")

if avg_survivors < 3:
    print("   ⚠️  Low survival rate - consider easier track or better sensors")
elif avg_survivors > 7:
    print("   ✅ High survival rate - your AI is getting good!")

# Plateau detection
if len(df) >= 10:
    recent_10 = df.tail(10)
    improvement_variance = recent_10['performance_improvement'].var()
    
    if improvement_variance < 50:  # Low variance in improvements
        print(f"\n🔄 PLATEAU DETECTED:")
        print(f"   - Low improvement variance in recent generations")
        print(f"   - Consider: Higher mutation rates or architecture changes")
    else:
        print(f"\n🎢 ACTIVE LEARNING:")
        print(f"   - High variance in improvements - still exploring")

# Next steps recommendations
print(f"\n🎯 NEXT STEPS RECOMMENDATIONS:")
print(f"   1. Continue training with mutation rate: {best_tested_rate:.3f}")

if len(unique_rates) == 1:
    print(f"   2. Experiment with different rates: Try 0.05, 0.1, 0.15, 0.2")
else:
    untested_rates = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
    suggested_rates = [r for r in untested_rates if r not in unique_rates][:3]
    if suggested_rates:
        print(f"   2. Try untested rates: {', '.join([f'{r:.3f}' for r in suggested_rates])}")

print(f"   3. Run at least 10 more generations for better analysis")
print(f"   4. Consider adaptive mutation (decrease rate as performance improves)")

# Success metrics
success_threshold = df['best_fitness'].quantile(0.8)  # Top 20% performance
successful_gens = df[df['best_fitness'] >= success_threshold]

if len(successful_gens) > 0:
    successful_rates = successful_gens['mutation_rate'].value_counts()
    print(f"\n🏅 SUCCESS PATTERN:")
    print(f"   - {len(successful_gens)} high-performing generations out of {len(df)}")
    print(f"   - Most successful mutation rate: {successful_rates.index[0]:.3f}")
    print(f"     (used in {successful_rates.iloc[0]} high-performance generations)")

print("\n" + "="*80)
print("💾 Analysis saved as: mutation_impact_analysis_REAL.png")
print("🔬 Run more generations and press 'M' again for updated insights!")
print("="*80)