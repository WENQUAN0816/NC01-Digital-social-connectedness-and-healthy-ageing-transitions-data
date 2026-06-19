"""
XGBoost + SHAP Analysis for npj Digital Medicine submission
完整的机器学习分析：训练、验证、SHAP解释、公平性评估
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# ML libraries
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import shap

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class HealthyAgingMLAnalysis:
    """完整的ML分析流程"""

    def __init__(self, data_path, output_dir):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)

        self.results = {}

    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("=" * 80)
        print("Step 1: Loading and preparing data...")
        print("=" * 80)

        # 尝试从多个可能的位置加载数据
        possible_paths = [
            self.data_path,
            Path("F:/目前养老官方数据库/七国包括charls（非常简单明了的数据）/charls.csv"),
        ]

        df = None
        for path in possible_paths:
            if path.exists():
                print(f"Loading from: {path}")
                if path.suffix == '.parquet':
                    df = pd.read_parquet(path)
                elif path.suffix == '.csv':
                    df = pd.read_csv(path, low_memory=False)
                break

        if df is None:
            raise FileNotFoundError(f"Cannot find data file in: {possible_paths}")

        print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")

        # 数据质量报告
        print("\n" + "=" * 80)
        print("DATA QUALITY REPORT")
        print("=" * 80)

        # 检查关键变量
        key_vars = [
            'internet_use', 'social_participation', 'healthy_aging_binary',
            'age', 'female', 'education_group', 'cognition_z',
            'depressive_symptoms', 'adl_limitation', 'multimorbidity'
        ]

        missing_vars = [v for v in key_vars if v not in df.columns]
        if missing_vars:
            print(f"⚠ Missing variables: {missing_vars}")
            # 尝试映射常见的变量名
            var_mapping = {
                'internet_use': ['social10', 'internet'],
                'social_participation': ['socwk', 'group6', 'social'],
                'healthy_aging_binary': ['healthy_aging', 'frailtyb'],
                'cognition_z': ['tcog_z_z', 'cognition'],
                'depressive_symptoms': ['cesd10', 'cesd', 'depression'],
            }

            for standard_name, alternatives in var_mapping.items():
                if standard_name not in df.columns:
                    for alt in alternatives:
                        if alt in df.columns:
                            df[standard_name] = df[alt]
                            print(f"  Mapped {alt} -> {standard_name}")
                            break

        # 缺失值统计
        print("\nMissing Value Summary:")
        for var in key_vars:
            if var in df.columns:
                missing_pct = df[var].isna().mean() * 100
                print(f"  {var:30s}: {missing_pct:5.1f}% missing")

        # 变量分布
        print("\nVariable Distribution:")
        for var in ['internet_use', 'social_participation', 'healthy_aging_binary']:
            if var in df.columns and df[var].notna().any():
                value_counts = df[var].value_counts(dropna=False)
                print(f"\n  {var}:")
                for val, count in value_counts.items():
                    pct = count / len(df) * 100
                    print(f"    {val}: {count:,} ({pct:.1f}%)")

        self.df_raw = df
        return df

    def create_analysis_dataset(self, cohort_filter=None):
        """创建分析数据集"""
        print("\n" + "=" * 80)
        print("Step 2: Creating analysis dataset...")
        print("=" * 80)

        df = self.df_raw.copy()

        # 如果指定队列筛选
        if cohort_filter and 'cohort' in df.columns:
            df = df[df['cohort'].isin(cohort_filter)]
            print(f"Filtered to cohorts: {cohort_filter}")

        # 定义特征
        feature_cols = [
            'internet_use', 'social_participation',
            'age', 'female',
            'cognition_z', 'depressive_symptoms',
            'adl_limitation', 'multimorbidity'
        ]

        # 添加教育和城乡（如果是分类变量，需要编码）
        if 'education_group' in df.columns:
            # 将教育分组编码为数值
            edu_map = {'low': 0, 'medium': 1, 'high': 2, 'unknown': -1}
            df['education_numeric'] = df['education_group'].map(edu_map)
            if df['education_numeric'].notna().any():
                feature_cols.append('education_numeric')

        if 'rural_category' in df.columns:
            rural_map = {'urban': 0, 'rural': 1, 'unknown': -1}
            df['rural_numeric'] = df['rural_category'].map(rural_map)
            if df['rural_numeric'].notna().any():
                feature_cols.append('rural_numeric')

        # 添加婚姻状态
        if 'married' in df.columns:
            feature_cols.append('married')

        # 检查哪些特征实际存在
        available_features = [f for f in feature_cols if f in df.columns]
        missing_features = [f for f in feature_cols if f not in available_features]

        print(f"\n✓ Available features ({len(available_features)}): {available_features}")
        if missing_features:
            print(f"⚠ Missing features ({len(missing_features)}): {missing_features}")

        # 目标变量
        target_col = 'healthy_aging_binary'

        # 删除缺失值
        analysis_cols = available_features + [target_col]
        if 'cohort' in df.columns:
            analysis_cols.append('cohort')
        if 'panel_id' in df.columns:
            analysis_cols.append('panel_id')

        df_clean = df[analysis_cols].dropna()

        print(f"\nSample size:")
        print(f"  Before cleaning: {len(df):,}")
        print(f"  After cleaning:  {len(df_clean):,}")
        print(f"  Loss:            {len(df) - len(df_clean):,} ({(len(df) - len(df_clean)) / len(df) * 100:.1f}%)")

        # 目标变量分布
        if target_col in df_clean.columns:
            target_dist = df_clean[target_col].value_counts()
            print(f"\nTarget variable ({target_col}) distribution:")
            for val, count in target_dist.items():
                pct = count / len(df_clean) * 100
                print(f"  {val}: {count:,} ({pct:.1f}%)")

        self.df_analysis = df_clean
        self.feature_cols = available_features
        self.target_col = target_col

        return df_clean, available_features, target_col

    def train_xgboost_model(self, X, y, cohort=None):
        """训练 XGBoost 模型 + 5-fold CV"""
        print("\n" + "=" * 80)
        print("Step 3: Training XGBoost model...")
        print("=" * 80)

        # 5-fold 交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_results = []
        models = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            print(f"\nFold {fold}/5...")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 训练模型
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold,
                eval_metric='logloss'
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # 预测
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # 评估
            auc = roc_auc_score(y_val, y_pred_proba)
            acc = accuracy_score(y_val, y_pred)
            brier = brier_score_loss(y_val, y_pred_proba)

            print(f"  AUC: {auc:.4f}, Accuracy: {acc:.4f}, Brier: {brier:.4f}")

            cv_results.append({
                'fold': fold,
                'auc': auc,
                'accuracy': acc,
                'brier': brier,
                'n_train': len(X_train),
                'n_val': len(X_val)
            })

            models.append(model)

        # 汇总
        cv_df = pd.DataFrame(cv_results)
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Mean AUC:      {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
        print(f"Mean Accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
        print(f"Mean Brier:    {cv_df['brier'].mean():.4f} ± {cv_df['brier'].std():.4f}")

        # 保存结果
        cv_df.to_csv(self.output_dir / "tables" / "cv_results.csv", index=False)

        # 在全部数据上训练最终模型
        print("\nTraining final model on full dataset...")
        final_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        final_model.fit(X, y, verbose=False)

        self.model = final_model
        self.cv_results = cv_df

        return final_model, cv_df

    def compute_shap_values(self, X, sample_size=1000):
        """计算 SHAP 值"""
        print("\n" + "=" * 80)
        print("Step 4: Computing SHAP values...")
        print("=" * 80)

        # 如果样本太大，采样以加速
        if len(X) > sample_size:
            print(f"Sampling {sample_size} from {len(X)} for SHAP analysis...")
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X

        print("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(self.model)

        print("Computing SHAP values...")
        shap_values = explainer.shap_values(X_sample)

        print("✓ SHAP computation complete")

        self.explainer = explainer
        self.shap_values = shap_values
        self.X_shap = X_sample

        return shap_values, X_sample

    def plot_shap_summary(self):
        """绘制 SHAP summary plot"""
        print("\nGenerating SHAP summary plot...")

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_shap,
            show=False,
            max_display=15
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "shap_summary.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "figures" / "shap_summary.pdf", bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: shap_summary.png/pdf")

    def plot_shap_dependence(self, features=None):
        """绘制 SHAP dependence plots"""
        if features is None:
            # 默认绘制最重要的3个特征
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            top_features = self.X_shap.columns[np.argsort(mean_abs_shap)[::-1][:3]]
        else:
            top_features = features

        print(f"\nGenerating SHAP dependence plots for: {list(top_features)}")

        for feature in top_features:
            if feature not in self.X_shap.columns:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            shap.dependence_plot(
                feature,
                self.shap_values,
                self.X_shap,
                show=False,
                ax=ax
            )
            plt.tight_layout()

            safe_name = feature.replace('/', '_').replace(' ', '_')
            plt.savefig(self.output_dir / "figures" / f"shap_dependence_{safe_name}.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ {feature}")

    def analyze_subgroup_heterogeneity(self):
        """分析亚组异质性"""
        print("\n" + "=" * 80)
        print("Step 5: Subgroup heterogeneity analysis...")
        print("=" * 80)

        df = self.df_analysis

        # 定义亚组
        subgroups = []

        # 年龄
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 70, 100], labels=['<70', '≥70'])
            subgroups.append(('age_group', ['<70', '≥70']))

        # 性别
        if 'female' in df.columns:
            df['sex_group'] = df['female'].map({0: 'male', 1: 'female'})
            subgroups.append(('sex_group', ['male', 'female']))

        # 教育
        if 'education_group' in df.columns:
            subgroups.append(('education_group', ['low', 'medium', 'high']))

        # 城乡
        if 'rural_category' in df.columns:
            subgroups.append(('rural_category', ['urban', 'rural']))

        # 计算每个亚组的 SHAP 均值
        results = []

        for group_var, group_vals in subgroups:
            if group_var not in df.columns:
                continue

            print(f"\nAnalyzing {group_var}...")

            for val in group_vals:
                # 获取该亚组的索引
                mask = df[group_var] == val
                if not mask.any():
                    continue

                # 获取该亚组在 SHAP 样本中的索引
                shap_mask = self.X_shap.index.isin(df[mask].index)
                if not shap_mask.any():
                    continue

                # 计算该亚组的平均 SHAP 值
                mean_shap = self.shap_values[shap_mask].mean(axis=0)

                for i, feature in enumerate(self.X_shap.columns):
                    results.append({
                        'subgroup_variable': group_var,
                        'subgroup_value': val,
                        'feature': feature,
                        'mean_shap': mean_shap[i],
                        'n': shap_mask.sum()
                    })

                print(f"  {val}: n={shap_mask.sum()}")

        results_df = pd.DataFrame(results)

        # 保存
        results_df.to_csv(self.output_dir / "tables" / "subgroup_shap.csv", index=False)

        # 可视化：对比关键特征在不同亚组的重要性
        self._plot_subgroup_comparison(results_df)

        print(f"\n✓ Saved: subgroup_shap.csv")

        return results_df

    def _plot_subgroup_comparison(self, results_df):
        """绘制亚组对比图"""
        # 选择 internet_use 和 social_participation 的 SHAP 值
        key_features = ['internet_use', 'social_participation']
        available_key_features = [f for f in key_features if f in self.X_shap.columns]

        if not available_key_features:
            return

        plot_df = results_df[results_df['feature'].isin(available_key_features)]

        if len(plot_df) == 0:
            return

        # 按 subgroup_variable 分组绘图
        for subgroup_var in plot_df['subgroup_variable'].unique():
            subset = plot_df[plot_df['subgroup_variable'] == subgroup_var]

            fig, ax = plt.subplots(figsize=(10, 6))

            # Pivot for grouped bar chart
            pivot = subset.pivot(index='subgroup_value', columns='feature', values='mean_shap')
            pivot.plot(kind='bar', ax=ax, width=0.8)

            ax.set_title(f'SHAP Values by {subgroup_var}', fontsize=14, fontweight='bold')
            ax.set_xlabel(subgroup_var, fontsize=12)
            ax.set_ylabel('Mean SHAP Value', fontsize=12)
            ax.legend(title='Feature', fontsize=10)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)
            plt.xticks(rotation=0)
            plt.tight_layout()

            safe_name = subgroup_var.replace('/', '_').replace(' ', '_')
            plt.savefig(self.output_dir / "figures" / f"subgroup_comparison_{safe_name}.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  ✓ Plotted: {subgroup_var}")

    def external_validation(self, cohorts):
        """外部验证"""
        if 'cohort' not in self.df_analysis.columns:
            print("\n⚠ No cohort column found, skipping external validation")
            return None

        print("\n" + "=" * 80)
        print("Step 6: External validation...")
        print("=" * 80)

        # 在 CHARLS 上训练
        train_cohort = 'CHARLS'
        train_data = self.df_analysis[self.df_analysis['cohort'] == train_cohort]

        if len(train_data) == 0:
            print(f"⚠ No data for training cohort {train_cohort}")
            return None

        X_train = train_data[self.feature_cols]
        y_train = train_data[self.target_col]

        print(f"Training on {train_cohort}: n={len(train_data)}")

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X_train, y_train, verbose=False)

        # 在其他队列验证
        val_cohorts = [c for c in cohorts if c != train_cohort and c in self.df_analysis['cohort'].unique()]

        results = []

        for cohort in val_cohorts:
            val_data = self.df_analysis[self.df_analysis['cohort'] == cohort]

            if len(val_data) == 0:
                continue

            X_val = val_data[self.feature_cols]
            y_val = val_data[self.target_col]

            # 预测
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # 评估
            auc = roc_auc_score(y_val, y_pred_proba)
            brier = brier_score_loss(y_val, y_pred_proba)

            print(f"  {cohort}: n={len(val_data)}, AUC={auc:.4f}, Brier={brier:.4f}")

            results.append({
                'cohort': cohort,
                'n': len(val_data),
                'auc': auc,
                'brier': brier
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "tables" / "external_validation.csv", index=False)

        print(f"\n✓ Saved: external_validation.csv")

        return results_df

    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)

        report = []
        report.append("# XGBoost + SHAP Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 数据概览
        report.append("## Data Overview")
        report.append(f"- Total samples: {len(self.df_analysis):,}")
        report.append(f"- Features: {len(self.feature_cols)}")
        report.append(f"- Feature list: {', '.join(self.feature_cols)}")
        report.append("")

        # CV 结果
        report.append("## Cross-Validation Performance")
        report.append(f"- Mean AUC: {self.cv_results['auc'].mean():.4f} ± {self.cv_results['auc'].std():.4f}")
        report.append(f"- Mean Accuracy: {self.cv_results['accuracy'].mean():.4f} ± {self.cv_results['accuracy'].std():.4f}")
        report.append(f"- Mean Brier Score: {self.cv_results['brier'].mean():.4f} ± {self.cv_results['brier'].std():.4f}")
        report.append("")

        # Feature importance
        report.append("## Feature Importance (SHAP)")
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.X_shap.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)

        report.append("Top 10 features:")
        for i, row in feature_importance.head(10).iterrows():
            report.append(f"  {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")
        report.append("")

        # 输出文件列表
        report.append("## Output Files")
        report.append("### Tables")
        for f in sorted((self.output_dir / "tables").glob("*.csv")):
            report.append(f"  - {f.name}")
        report.append("")
        report.append("### Figures")
        for f in sorted((self.output_dir / "figures").glob("*.png")):
            report.append(f"  - {f.name}")

        report_text = "\n".join(report)

        # 保存
        report_path = self.output_dir / "ANALYSIS_REPORT.md"
        report_path.write_text(report_text, encoding='utf-8')

        print(f"\n✓ Report saved: {report_path}")
        print("\n" + "=" * 80)
        print(report_text)
        print("=" * 80)

        return report_text


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='XGBoost + SHAP Analysis for npj Digital Medicine')
    parser.add_argument('--data', type=str,
                       default='F:/目前养老官方数据库/七国包括charls（非常简单明了的数据）/charls.csv',
                       help='Path to data file')
    parser.add_argument('--output', type=str,
                       default='./ml_analysis_output',
                       help='Output directory')
    parser.add_argument('--cohorts', nargs='+', default=['CHARLS', 'HRS', 'ELSA', 'SHARE'],
                       help='Cohorts for external validation')

    args = parser.parse_args()

    # 运行分析
    analyzer = HealthyAgingMLAnalysis(args.data, args.output)

    # Step 1: Load data
    analyzer.load_and_prepare_data()

    # Step 2: Prepare analysis dataset
    df_clean, features, target = analyzer.create_analysis_dataset(cohort_filter=None)

    # Step 3: Train XGBoost
    X = df_clean[features]
    y = df_clean[target]
    model, cv_results = analyzer.train_xgboost_model(X, y)

    # Step 4: SHAP analysis
    shap_values, X_shap = analyzer.compute_shap_values(X)
    analyzer.plot_shap_summary()
    analyzer.plot_shap_dependence()

    # Step 5: Subgroup analysis
    analyzer.analyze_subgroup_heterogeneity()

    # Step 6: External validation
    if 'cohort' in df_clean.columns:
        analyzer.external_validation(args.cohorts)

    # Step 7: Summary report
    analyzer.generate_summary_report()

    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print(f"All results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
