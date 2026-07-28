import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.base import BaseEstimator, TransformerMixin

#---------------------------------
# A dict of shortened value names
#---------------------------------
short_names = {
    'Thyroid Function': {
        'Euthyroid': 'Euthy',
        'Clinical Hyperthyroidism': 'Hyper',
        'Clinical Hypothyroidism': 'Hypo',
        'Subclinical Hyperthyroidism': 'SubHyper',
        'Subclinical Hypothyroidism': 'SubHypo'
    },
    'Physical Examination': {
        'Single nodular goiter-left': 'SingleL',
        'Multinodular goiter': 'Multi',
        'Single nodular goiter-right': 'SingleR',
        'Normal': 'Normal',
        'Diffuse goiter': 'Diffuse'
    },
    'Adenopathy': {
        'No': 'No',
        'Right': 'R',
        'Extensive': 'Ext',
        'Left': 'L',
        'Bilateral': 'Bil',
        'Posterior': 'Post'
    },
    'Pathology': {
        'Micropapillary': 'Micro',
        'Papillary': 'Papp',
        'Follicular': 'Fol',
        'Hurthel cell': 'Hurth'
    },
    'Focality': {'Uni-Focal': 'Uni', 'Multi-Focal': 'Multi'},
    'Risk': {'Low': 'Low', 'Intermediate': 'Int', 'High': 'High'},
    'T': {'T1a': 'T1a', 'T1b': 'T1b', 'T2': 'T2', 'T3a': 'T3a', 'T3b': 'T3b', 'T4a': 'T4a', 'T4b': 'T4b'},
    'N': {'N0': 'N0', 'N1b': 'N1b', 'N1a': 'N1a'},
    'M': {'M0': 'M0', 'M1': 'M1'},
    'Stage': {'I': 'I', 'II': 'II', 'IVB': 'IVB', 'III': 'III', 'IVA': 'IVA'},
    'Response': {'Indeterminate': 'Indet', 'Excellent': 'Exc', 'Structural Incomplete': 'Struct', 'Biochemical Incomplete': 'Bio'},
    'Recurred': {'No': 'No', 'Yes': 'Yes'}
}

rename_dict = {
    # Continuous
    "con_num__Age": "Age",

    # Binary
    "bin__Gender_F": "Female",
    "bin__Gender_M": "Male",
    "bin__Smoking_No": "Smoke_No",
    "bin__Smoking_Yes": "Smoke_Yes",
    "bin__Hx Smoking_No": "HxSmoke_No",
    "bin__Hx Smoking_Yes": "HxSmoke_Yes",
    "bin__Hx Radiothreapy_No": "HxRadio_No",
    "bin__Hx Radiothreapy_Yes": "HxRadio_Yes",
    "bin__Focality_Uni-Focal": "UniFocal",
    "bin__Focality_Multi-Focal": "MultiFocal",

    # Thyroid Function
    "nom__Thyroid Function_Clinical Hyperthyroidism": "ClinHyper",
    "nom__Thyroid Function_Clinical Hypothyroidism": "ClinHypo",
    "nom__Thyroid Function_Euthyroid": "Euthy",
    "nom__Thyroid Function_Subclinical Hyperthyroidism": "SubHyper",
    "nom__Thyroid Function_Subclinical Hypothyroidism": "SubHypo",

    # Physical Examination
    "nom__Physical Examination_Diffuse goiter": "Diffuse",
    "nom__Physical Examination_Multinodular goiter": "MultiGoiter",
    "nom__Physical Examination_Normal": "Normal",
    "nom__Physical Examination_Single nodular goiter-left": "SingleLeft",
    "nom__Physical Examination_Single nodular goiter-right": "SingleRight",

    # Adenopathy
    "nom__Adenopathy_Bilateral": "Bilateral",
    "nom__Adenopathy_Extensive": "Extensive",
    "nom__Adenopathy_Left": "Left",
    "nom__Adenopathy_No": "NoAden",
    "nom__Adenopathy_Posterior": "Posterior",
    "nom__Adenopathy_Right": "Right",

    # Pathology
    "nom__Pathology_Follicular": "Follicular",
    "nom__Pathology_Hurthel cell": "Hurthel",
    "nom__Pathology_Micropapillary": "MicroPap",
    "nom__Pathology_Papillary": "Papillary",

    # Ordinal
    "ord__Risk": "Risk",
    "ord__T": "T",
    "ord__N": "N",
    "ord__M": "M",
    "ord__Stage": "Stage",
    "ord__Response": "Response"
}

#----------------
# Check skewness
#----------------
def skewness_detector(num_cols):
    # Compute skewness for each numerical features and add to dict
    sk = {col: abs(num_cols[col].skew()) for col in num_cols}
    # Convert to dataframe
    skewness = pd.DataFrame(sk, index=[0])
    # Melt (for better inspection)
    skewness = pd.melt(skewness, var_name='Feature', value_name='Skewness')
    # Check if skewness is exceeding the threshold (skewness >= 1)
    skewness['Skewness_Exceeding_Threshold'] = skewness['Skewness'] >= 1
    return skewness.sort_values(by='Skewness', ascending=False).reset_index(drop=True)


#-------------------------------------------
# Find potential outliers in numeric values
#-------------------------------------------
def outlier_detector(num_cols):
    # IQR method (1.5 * IQR) 
    outlier_info = [] 
    # Loop through numeric features
    for c in num_cols: 
        # Calculate Q1 -> 25%
        Q1 = num_cols[c].quantile(0.25) 
        # Calculate Q3 -> 75%
        Q3 = num_cols[c].quantile(0.75) 
        # Calculate IQR
        IQR = Q3 - Q1 
        # Calculate and determine lower and upper bounds
        lower = Q1 - 1.5 * IQR 
        upper = Q3 + 1.5 * IQR 
        # Filter outliers
        mask = (num_cols[c] < lower) | (num_cols[c] > upper) 
        # Count outliers
        n_out = mask.sum() 
        # Add all info to list (features, number of outliers, percentage of outliers)
        outlier_info.append((c, n_out, (n_out / len(num_cols) * 100).round(3)))

    # Convert to dataframe
    outlier_df = pd.DataFrame(outlier_info, columns=['Feature', 'N_outliers', 'Outlier_pct'])
    return outlier_df.sort_values(by="Outlier_pct", ascending=False).reset_index(drop=True)

#--------------------------------------------------
# Calculate numeric features proportions in x bins
#--------------------------------------------------
def numeric_feat_propor_calc(df, num_bin=30):
    for col in df:
        bins = pd.cut(df[col], bins=num_bin)
        proportions = bins.value_counts(normalize=True).sort_index()

        # Display as DataFrame
        prop_df = proportions.reset_index()
        prop_df.columns = [f'{col} Range', 'Proportion']
        print(prop_df, end="\n\n")


#-----------------------------------------------------------------
# Correlation class for all types of features
#-----------------------------------------------------------------
class CorrelationAnalyzer:
    """
    Plot various association heatmaps for a dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The full dataset (or training set).
    num_cols : list of str
        Names of numeric columns.
    cat_cols : list of str
        Names of categorical columns.
    target_col : str, optional
        Name of the target column for feature‑target heatmaps.
    """

    def __init__(self, data, target_type=None, num_cols=None, cat_cols=None, target_col=None):

        self.data = data
        self.num_cols = list(num_cols) if num_cols is not None else []
        self.cat_cols = list(cat_cols) if cat_cols is not None else []
        self.target_col = target_col
        if target_type is not None:
            self.target_type = target_type.lower()
        else:
            self.target_type = target_type

        self.num_num_corr = None
        self.cat_cat_corr = None
        self.num_cat_eta = None
        self.num_target_corr = None
        self.cat_target_corr = None

        

    # ---------- Static helpers ----------
    @staticmethod
    def _cramers_v(x, y):
        """Cramér's V between two categorical series."""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    @staticmethod
    def _correlation_ratio(categories, values):
        """η² (eta‑squared) root: association between categorical and numeric."""
        categories = pd.Series(categories)
        values = pd.Series(values)
        grand_mean = values.mean()
        ss_total = ((values - grand_mean) ** 2).sum()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2
            for _, g in values.groupby(categories)
        )
        return np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0

    # ---------- Plotting methods ----------
    def plot_num_num(self, figsize=(8, 6)):
        """Pearson correlation heatmap for numeric × numeric."""
        corr = self.data[self.num_cols].corr(method='pearson')

        # Only sort if one dimension is 1 (i.e., 1xN or Nx1)
        if corr.shape[0] == 1 or corr.shape[1] == 1:
            if corr.shape[0] == 1:        # one row → sort columns
                col_order = corr.iloc[0].abs().sort_values(ascending=False).index
                corr = corr.loc[corr.index, col_order]   # keep row index, reorder columns
            else:                         # one column → sort rows
                row_order = corr.iloc[:, 0].abs().sort_values(ascending=False).index
                corr = corr.loc[row_order, corr.columns] # keep column names, reorder rows

        self.num_num_corr = corr

        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                    linewidths=0.5, square=True)
        plt.title('Numeric ↔ Numeric (Pearson r)')
        plt.tight_layout()
        plt.show()

    def plot_cat_cat(self, figsize=(8, 6)):
        """Cramér's V heatmap for categorical × categorical."""
        cramer = pd.DataFrame(index=self.cat_cols, columns=self.cat_cols, dtype=float)
        for c1 in self.cat_cols:
            for c2 in self.cat_cols:
                cramer.loc[c1, c2] = self._cramers_v(self.data[c1], self.data[c2])

        # Sort Cramér's V
        if cramer.shape[0] == 1 or cramer.shape[1] == 1:
            if cramer.shape[0] == 1:
                col_order = cramer.iloc[0].sort_values(ascending=False).index
                cramer = cramer.loc[cramer.index, col_order]
            else:
                row_order = cramer.iloc[:, 0].sort_values(ascending=False).index
                cramer = cramer.loc[row_order, cramer.columns]

        self.cat_cat_corr = cramer

        plt.figure(figsize=figsize)
        sns.heatmap(cramer.astype(float), annot=True, cmap='YlGnBu',
                    vmin=0, vmax=1, linewidths=0.5, square=True)
        plt.title('Categorical ↔ Categorical (Cramér\'s V)')
        plt.tight_layout()
        plt.show()

    def plot_num_cat(self, figsize=(10, 4)):
        """Correlation ratio (η) heatmap for numeric × categorical."""
        eta = pd.DataFrame(index=self.num_cols, columns=self.cat_cols, dtype=float)
        for n in self.num_cols:
            for c in self.cat_cols:
                eta.loc[n, c] = self._correlation_ratio(self.data[c], self.data[n])

        # Sort
        if eta.shape[0] == 1 or eta.shape[1] == 1:
            if eta.shape[0] == 1:         # one numeric → sort categories (columns)
                col_order = eta.iloc[0].sort_values(ascending=False).index
                eta = eta.loc[eta.index, col_order]
            else:                         # one categorical → sort numerics (rows)
                row_order = eta.iloc[:, 0].sort_values(ascending=False).index
                eta = eta.loc[row_order, eta.columns]

        self.num_cat_eta = eta

        plt.figure(figsize=figsize)
        sns.heatmap(eta.astype(float), annot=True, cmap='YlOrRd',
                    vmin=0, vmax=1, linewidths=0.5)
        plt.title('Numeric ↔ Categorical (Correlation Ratio η)')
        plt.ylabel('Numeric Features')
        plt.xlabel('Categorical Features')
        plt.tight_layout()
        plt.show()

    def plot_feature_categorical_target(self):
        """Feature ↔ target: point-biserial (numeric) and Cramér's V (categorical).
            - Only for categorical targets
        """
        if self.target_col is None:
            raise ValueError("target_col must be set to plot feature‑target heatmaps.")
        target = self.data[self.target_col]

        fig, axes = plt.subplots(2, 1, figsize=(12, 5),
                                 gridspec_kw={'height_ratios': [1, 1]})

        # Numeric → target (point‑biserial)
        pb = {c: pointbiserialr(self.data[c], target)[0] for c in self.num_cols}
        pb_df = pd.DataFrame(pb, index=['r']).sort_values(by='r', axis=1, ascending=False)
        self.num_target_corr = pb_df 

        sns.heatmap(pb_df, annot=True, cmap='coolwarm', center=0, ax=axes[0])
        axes[0].set_title('Numeric → Target (Point‑biserial r)')

        # Categorical → target (Cramér's V)
        cv = {c: self._cramers_v(self.data[c], target) for c in self.cat_cols}
        cv_df = pd.DataFrame(cv, index=['V']).sort_values(by='V', axis=1, ascending=False)
        self.cat_target_corr = cv_df 

        sns.heatmap(cv_df, annot=True, cmap='YlGnBu', vmin=0, vmax=1, ax=axes[1])
        axes[1].set_title('Categorical → Target (Cramér\'s V)')

        plt.tight_layout()
        plt.show()

    def plot_feature_numeric_target(self):
        """
        Feature ↔ Target associations.

        Numeric features:
            Pearson correlation (continuous target)

        Categorical features:
            Correlation ratio (η)
        """

        if self.target_col is None:
            raise ValueError("target_col must be set to plot feature-target heatmaps.")

        target = self.data[self.target_col]
        n_plots = int(bool(self.num_cols)) + int(bool(self.cat_cols))

        if n_plots == 0:
            raise ValueError("At least one of num_cols or cat_cols must be provided.")

        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots))

        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # -------------------------------------------------
        # Numeric -> Continuous Target (Pearson)
        # -------------------------------------------------
        if self.num_cols:

            pearson_corr = {col: self.data[col].corr(target) for col in self.num_cols}

            pearson_df = (pd.DataFrame(pearson_corr, index=['r']).sort_values(by='r', axis=1, ascending=False))
            self.num_target_corr = pearson_df

            sns.heatmap(pearson_df, annot=True, cmap='coolwarm', center=0, ax=axes[plot_idx])

            axes[plot_idx].set_title('Numeric → Target (Pearson r)')
            plot_idx += 1

        # -------------------------------------------------
        # Categorical -> Continuous Target (Correlation Ratio η)
        # -------------------------------------------------
        if self.cat_cols:
            eta_corr = {col: self._correlation_ratio(self.data[col], target) for col in self.cat_cols}
            eta_df = (pd.DataFrame(eta_corr, index=['η']).sort_values(by='η', axis=1, ascending=False))

            self.cat_target_corr = eta_df

            sns.heatmap(eta_df, annot=True, cmap='YlGnBu', vmin=0, vmax=1, ax=axes[plot_idx])
            axes[plot_idx].set_title('Categorical → Target (Correlation Ratio η)')

        plt.tight_layout()
        plt.show()

    def plot_all(self, include_target=True):
        """Run all four plots (optionally with target)."""
        self.plot_num_num()
        self.plot_cat_cat()
        self.plot_num_cat()
        if include_target and self.target_col is not None:
            if self.target_type == 'numeric':
                self.plot_feature_numeric_target()
            elif self.target_type == 'categorical':
                self.plot_feature_categorical_target()

    def get_correlation_results(self):
        """Return a dictionary of all computed correlation matrices."""
        return {
            'num_num': self.num_num_corr,
            'cat_cat': self.cat_cat_corr,
            'num_cat': self.num_cat_eta,
            'num_target': self.num_target_corr,
            'cat_target': self.cat_target_corr
        }


