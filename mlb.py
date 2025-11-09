import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load the data
df = pd.read_csv('mlb_payroll.csv')

# Data cleaning - Remove $ and commas, convert to numeric
def clean_currency(column):
    return pd.to_numeric(column.astype(str).str.replace('[$,]', '', regex=True), errors='coerce')

df['Total Payroll'] = clean_currency(df['Total Payroll'])
df['Active'] = clean_currency(df['Active'])
df['Injured'] = clean_currency(df['Injured'])
df['Retained'] = clean_currency(df['Retained'])
df['Buried'] = clean_currency(df['Buried'])

df['Total_Payroll_Millions'] = df['Total Payroll'] / 1_000_000
df['Active_Payroll_Millions'] = df['Active'] / 1_000_000

# Extract wins and losses from Record column
df = df.dropna(subset=['Record'])
df[['Wins', 'Losses']] = df['Record'].str.split('-', expand=True).astype(int)
df['Win_Percentage'] = df['Wins'] / (df['Wins'] + df['Losses'])

# Calculate payroll efficiency (wins per million dollars)
df['Efficiency'] = df['Wins'] / df['Total_Payroll_Millions']

print("MLB PAYROLL ANALYSIS - 2024 Season")
print(f"\nDataset Shape: {df.shape[0]} teams, {df.shape[1]} features")
print(f"\nAverage Total Payroll: ${df['Total_Payroll_Millions'].mean():.2f}M")
print(f"Median Total Payroll: ${df['Total_Payroll_Millions'].median():.2f}M")

# Key Statistics
print("\n")
print("KEY FINDINGS")

# Correlation
correlation = df['Total_Payroll_Millions'].corr(df['Wins'])
print(f"\nCorrelation between Total Payroll and Wins: {correlation:.3f}")

# Top spenders
print("\n--- Top 5 Spending Teams ---")
top_spenders = df.nlargest(5, 'Total_Payroll_Millions')[['Team', 'Total_Payroll_Millions', 'Wins', 'Record']]
print(top_spenders.to_string(index=False))

# Most efficient teams
print("\n--- Most Efficient Teams (Wins per $M) ---")
most_efficient = df.nlargest(5, 'Efficiency')[['Team', 'Total_Payroll_Millions', 'Wins', 'Efficiency']]
print(most_efficient.to_string(index=False))

# Least efficient teams
print("\n--- Least Efficient Teams (Wins per $M) ---")
least_efficient = df.nsmallest(5, 'Efficiency')[['Team', 'Total_Payroll_Millions', 'Wins', 'Efficiency']]
print(least_efficient.to_string(index=False))

# VISUALIZATION 1: Top 10 Team Payrolls
plt.figure(figsize=(14, 6))
top_10 = df.nlargest(10, 'Total_Payroll_Millions')
colors = ['#1f77b4' if x < 90 else '#2ca02c' for x in top_10['Wins']]
plt.bar(range(len(top_10)), top_10['Total_Payroll_Millions'], color=colors)
plt.xticks(range(len(top_10)), top_10['Team'], rotation=45, ha='right')
plt.ylabel('Total Payroll (Millions $)', fontsize=12)
plt.xlabel('Team', fontsize=12)
plt.title('Top 10 MLB Teams by Total Payroll - 2024 Season', fontsize=14, fontweight='bold')
plt.axhline(y=df['Total_Payroll_Millions'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'League Avg: ${df["Total_Payroll_Millions"].mean():.1f}M')
plt.legend()
plt.tight_layout()
plt.savefig('top_10_payrolls.png', dpi=300, bbox_inches='tight')
print("✓ Saved: top_10_payrolls.png")

# VISUALIZATION 2: Payroll vs Wins Scatter
plt.figure(figsize=(12, 7))
plt.scatter(df['Total_Payroll_Millions'], df['Wins'], 
           s=100, alpha=0.6, c=df['Win_Percentage'], cmap='RdYlGn')
plt.colorbar(label='Win Percentage')

# Add trend line
z = np.polyfit(df['Total_Payroll_Millions'], df['Wins'], 1)
p = np.poly1d(z)
plt.plot(df['Total_Payroll_Millions'], p(df['Total_Payroll_Millions']), 
         "r--", alpha=0.8, linewidth=2, label=f'Trend Line (R={correlation:.3f})')

plt.xlabel('Total Payroll (Millions $)', fontsize=12)
plt.ylabel('Wins', fontsize=12)
plt.title('MLB Team Payroll vs Performance - 2024 Season', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Annotate outliers
for idx, row in df.iterrows():
    if row['Efficiency'] > 0.45 or row['Efficiency'] < 0.25:
        plt.annotate(row['Team'], 
                    (row['Total_Payroll_Millions'], row['Wins']),
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('payroll_vs_wins.png', dpi=300, bbox_inches='tight')
print("✓ Saved: payroll_vs_wins.png")

# VISUALIZATION 3: Efficiency Analysis
plt.figure(figsize=(14, 6))
df_sorted = df.sort_values('Efficiency', ascending=True)
colors = ['#d62728' if x < 0.3 else '#2ca02c' if x > 0.4 else '#ff7f0e' 
          for x in df_sorted['Efficiency']]
plt.barh(range(len(df_sorted)), df_sorted['Efficiency'], color=colors)
plt.yticks(range(len(df_sorted)), df_sorted['Team'], fontsize=9)
plt.xlabel('Wins per Million Dollars Spent', fontsize=12)
plt.title('MLB Team Payroll Efficiency - 2024 Season', fontsize=14, fontweight='bold')
plt.axvline(x=df['Efficiency'].mean(), color='blue', linestyle='--', 
            linewidth=2, label=f'League Avg: {df["Efficiency"].mean():.3f}')
plt.legend()
plt.tight_layout()
plt.savefig('payroll_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: payroll_efficiency.png")

# VISUALIZATION 4: Payroll Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
ax1.hist(df['Total_Payroll_Millions'], bins=15, color='skyblue', edgecolor='black')
ax1.axvline(df['Total_Payroll_Millions'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Mean: ${df["Total_Payroll_Millions"].mean():.1f}M')
ax1.axvline(df['Total_Payroll_Millions'].median(), color='green', linestyle='--', 
            linewidth=2, label=f'Median: ${df["Total_Payroll_Millions"].median():.1f}M')
ax1.set_xlabel('Total Payroll (Millions $)', fontsize=11)
ax1.set_ylabel('Number of Teams', fontsize=11)
ax1.set_title('Distribution of MLB Team Payrolls', fontsize=12, fontweight='bold')
ax1.legend()

# Box plot
ax2.boxplot(df['Total_Payroll_Millions'], vert=True)
ax2.set_ylabel('Total Payroll (Millions $)', fontsize=11)
ax2.set_title('Payroll Distribution Box Plot', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('payroll_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: payroll_distribution.png")

# Export summary statistics to CSV
summary_stats = df[['Team', 'Record', 'Total_Payroll_Millions', 
                     'Wins', 'Win_Percentage', 'Efficiency']].sort_values('Wins', ascending=False)
summary_stats.to_csv('mlb_payroll_summary.csv', index=False)
print("✓ Saved: mlb_payroll_summary.csv")

print("\nFiles created:")
print("  1. top_10_payrolls.png - Bar chart of highest spending teams")
print("  2. payroll_vs_wins.png - Scatter plot showing payroll-performance relationship")
print("  3. payroll_efficiency.png - Teams ranked by cost efficiency")
print("  4. payroll_distribution.png - Statistical distribution of payrolls")
print("  5. mlb_payroll_summary.csv - Clean summary data")
