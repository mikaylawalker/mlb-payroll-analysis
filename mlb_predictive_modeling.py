import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("MLB PAYROLL PREDICTIVE MODELING - 2024 Season")

# Load the data
df = pd.read_csv('mlb_payroll.csv')

# Clean team names 
df['Team'] = df['Team'].str.strip()

# Feature Engineering
print("\nFeature Engineering...")

# Clean currency columns
currency_columns = ['Total Payroll', 'Active', 'Injured', 'Retained', 'Buried']
for col in currency_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df[col] = df[col].str.replace(' ', '', regex=False).str.replace('-', '0', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Create millions columns
df['Total_Payroll_Millions'] = df['Total Payroll'] / 1_000_000
df['Active_Payroll_Millions'] = df['Active'] / 1_000_000
df['Injured_Payroll_Millions'] = df['Injured'] / 1_000_000
df['Retained_Payroll_Millions'] = df['Retained'] / 1_000_000
df['Buried_Payroll_Millions'] = df['Buried'] / 1_000_000

# Extract wins and losses from Record
df[['Wins', 'Losses']] = df['Record'].str.split('-', expand=True)
df['Wins'] = pd.to_numeric(df['Wins'], errors='coerce')
df['Losses'] = pd.to_numeric(df['Losses'], errors='coerce')

# Remove any rows with missing data
df = df.dropna(subset=['Wins', 'Total_Payroll_Millions'])

df['Games_Played'] = df['Wins'] + df['Losses']
df['Win_Percentage'] = df['Wins'] / df['Games_Played']

# Create additional features
df['Payroll_Per_Win'] = df['Total_Payroll_Millions'] / df['Wins']
df['Active_Percentage'] = df['Active_Payroll_Millions'] / df['Total_Payroll_Millions']
df['Injured_Percentage'] = df['Injured_Payroll_Millions'] / df['Total_Payroll_Millions']

# Payroll categories for analysis
df['Payroll_Quartile'] = pd.qcut(df['Total_Payroll_Millions'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

print(f"   ✓ Created {len(df.columns)} total features")
print(f"   ✓ Dataset: {len(df)} teams")

# Define features and target
feature_columns = [
    'Total_Payroll_Millions',
    'Active_Payroll_Millions', 
    'Injured_Payroll_Millions',
    'Active_Percentage',
    'Injured_Percentage'
]

X = df[feature_columns]
y = df['Wins']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nData Split: {len(X_train)} training samples, {len(X_test)} test samples")

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
}

# Train and evaluate models
print("\nTraining Models...")
results = []

for name, model in models.items():
    # Train model
    if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Cross-validation on scaled data
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Cross-validation on original data
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'R² Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'CV R² Mean': cv_scores.mean(),
        'CV R² Std': cv_scores.std()
    })
    
    print(f"{name:20s} - R²: {r2:.3f}, RMSE: {rmse:.2f} wins")

results_df = pd.DataFrame(results).sort_values('R² Score', ascending=False)

print("\nModel Performance Summary")
print(results_df.to_string(index=False))

# Best model analysis
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"R² Score: {results_df.iloc[0]['R² Score']:.3f}")
print(f"RMSE: {results_df.iloc[0]['RMSE']:.2f} wins")
print(f"MAE: {results_df.iloc[0]['MAE']:.2f} wins")
print(f"Cross-Validation R²: {results_df.iloc[0]['CV R² Mean']:.3f} (±{results_df.iloc[0]['CV R² Std']:.3f})")

# Feature importance
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n   Feature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"      {row['Feature']:30s}: {row['Importance']:.3f}")

# Generate predictions for all teams using best model
print("Generate Full Dataset Predictions")
if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
    X_scaled = scaler.transform(X)
    df['Predicted_Wins'] = best_model.predict(X_scaled)
else:
    df['Predicted_Wins'] = best_model.predict(X)

df['Prediction_Error'] = df['Wins'] - df['Predicted_Wins']
df['Absolute_Error'] = abs(df['Prediction_Error'])

# Create comprehensive visualizations
fig = plt.figure(figsize=(18, 12))

# Model Comparison
ax1 = plt.subplot(2, 3, 1)
results_df_plot = results_df.sort_values('R² Score')
bars = ax1.barh(results_df_plot['Model'], results_df_plot['R² Score'], color='steelblue')
ax1.set_xlabel('R² Score', fontsize=11, fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold', pad=15)
ax1.set_xlim(0, 1)
for i, (bar, val) in enumerate(zip(bars, results_df_plot['R² Score'])):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
             va='center', fontsize=10, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Actual vs Predicted Wins
ax2 = plt.subplot(2, 3, 2)
scatter = ax2.scatter(df['Wins'], df['Predicted_Wins'], 
                     c=df['Total_Payroll_Millions'], cmap='RdYlGn', 
                     s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax2.plot([df['Wins'].min(), df['Wins'].max()], 
         [df['Wins'].min(), df['Wins'].max()], 
         'r--', linewidth=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Wins', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted Wins', fontsize=11, fontweight='bold')
ax2.set_title(f'Actual vs Predicted Wins\n({best_model_name})', fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Payroll ($M)', fontsize=10, fontweight='bold')

# Residual Plot
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(df['Predicted_Wins'], df['Prediction_Error'], 
           c=df['Total_Payroll_Millions'], cmap='RdYlGn',
           s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Predicted Wins', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residual (Actual - Predicted)', fontsize=11, fontweight='bold')
ax3.set_title('Residual Plot\n(Check for Bias)', fontsize=13, fontweight='bold', pad=15)
ax3.grid(alpha=0.3)

# RMSE Comparison
ax4 = plt.subplot(2, 3, 4)
results_df_plot = results_df.sort_values('RMSE', ascending=False)
bars = ax4.barh(results_df_plot['Model'], results_df_plot['RMSE'], color='coral')
ax4.set_xlabel('RMSE (Wins)', fontsize=11, fontweight='bold')
ax4.set_title('Model Error Comparison (RMSE)', fontsize=13, fontweight='bold', pad=15)
for i, (bar, val) in enumerate(zip(bars, results_df_plot['RMSE'])):
    ax4.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
             va='center', fontsize=10, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# Prediction Error Distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(df['Prediction_Error'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
ax5.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
ax5.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax5.set_title('Distribution of Prediction Errors', fontsize=13, fontweight='bold', pad=15)
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3)

# Top Over/Under Performers
ax6 = plt.subplot(2, 3, 6)
top_n = min(10, len(df))  # In case there are fewer than 10 teams
over_under = df.nlargest(top_n, 'Absolute_Error')[['Team', 'Prediction_Error']].sort_values('Prediction_Error')
colors = ['green' if x > 0 else 'red' for x in over_under['Prediction_Error']]
bars = ax6.barh(range(len(over_under)), over_under['Prediction_Error'], color=colors, alpha=0.7)
ax6.set_yticks(range(len(over_under)))
ax6.set_yticklabels(over_under['Team'], fontsize=9)
ax6.set_xlabel('Wins vs Prediction', fontsize=11, fontweight='bold')
ax6.set_title(f'Top {top_n} Over/Under Performers\n(Green = Better than predicted)', 
             fontsize=13, fontweight='bold', pad=15)
ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax6.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('mlb_predictive_model_analysis.png', dpi=300, bbox_inches='tight')

# Save predictions to CSV
output_df = df[['Team', 'Wins', 'Predicted_Wins', 'Prediction_Error', 
                'Absolute_Error', 'Total_Payroll_Millions', 'Payroll_Quartile']].sort_values('Absolute_Error', ascending=False)
output_df.to_csv('mlb_win_predictions.csv', index=False)

# Key Insights
print("\nMODEL PERFORMANCE:")
print(f"Best model explains {results_df.iloc[0]['R² Score']*100:.1f}% of variance in team wins")
print(f"Average prediction error: ±{results_df.iloc[0]['MAE']:.1f} wins")

best_team = df.nlargest(1, 'Prediction_Error').iloc[0]
worst_team = df.nsmallest(1, 'Prediction_Error').iloc[0]

print("\nBIGGEST OVERPERFORMERS:")
print(f"{best_team['Team']}: Won {best_team['Prediction_Error']:.1f} more games than predicted")
print(f"(Actual: {best_team['Wins']:.0f} wins, Predicted: {best_team['Predicted_Wins']:.1f})")

print("\nBIGGEST UNDERPERFORMERS:")
print(f"{worst_team['Team']}: Won {abs(worst_team['Prediction_Error']):.1f} fewer games than predicted")
print(f"(Actual: {worst_team['Wins']:.0f} wins, Predicted: {worst_team['Predicted_Wins']:.1f})")

# Payroll efficiency analysis
high_payroll = df[df['Payroll_Quartile'] == 'High']
low_payroll = df[df['Payroll_Quartile'] == 'Low']

print("\nPAYROLL VS PERFORMANCE:")
print(f"High payroll teams: {high_payroll['Wins'].mean():.1f} avg wins")
print(f"Low payroll teams: {low_payroll['Wins'].mean():.1f} avg wins")
print(f"Difference: {high_payroll['Wins'].mean() - low_payroll['Wins'].mean():.1f} wins")