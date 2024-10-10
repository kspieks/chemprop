import argparse
import os
import pandas as pd

n_models = 5

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('preds_path', type=str, help='The path to the prediction')
    parser.add_argument('rxn_column', type=str, help='The column name of the reaction SMILES')

    args = parser.parse_args()
    preds_path, rxn_column = args.preds_path, args.rxn_column

    # Read Gas and Solv results
    df_gas = pd.read_csv(preds_path + ".gas")
    df_solv = pd.read_csv(preds_path + ".solv")

    # Combine prediction
    df_gas = df_gas.rename(columns={rxn_column: f'{rxn_column}_gas'})
    df_pred = df_solv.merge(df_gas, how='left', left_on=rxn_column, right_on=f'{rxn_column}_gas')
    df_pred = pd.concat([df_solv, df_gas], axis=1)
    df_pred = df_pred.drop(columns=[f'{rxn_column}_gas'])

    # Compute solvated barrier heights
    for dg in ['DeltaG_ET', 'DeltaG_EP']:
        for i in range(n_models):
            df_pred[f'{dg}_soln_molar_kcal_mol_model_{i}'] = df_pred[f'{dg}_gas_molar_kcal_mol_model_{i}'] + df_pred[f'Delta{dg}_solv_molar_kcal_mol_model_{i}']
        df_pred[f'{dg}_soln_molar_kcal_mol'] = df_pred[[f'{dg}_soln_molar_kcal_mol_model_{i}' for i in range(n_models)]].mean(axis=1)
        df_pred[f'{dg}_soln_molar_kcal_mol_ensemble_uncal_var'] = df_pred[[f'{dg}_soln_molar_kcal_mol_model_{i}' for i in range(n_models)]].var(axis=1, ddof=0)

    # Reorder_columns
    new_columns = list(df_pred.columns[:2])
    for target in ["DeltaG_ET_gas", "DeltaG_EP_gas", "DeltaDeltaG_ET_solv", "DeltaDeltaG_EP_solv", "DeltaG_ET_soln", "DeltaG_EP_soln"]:
        new_columns += [target + "_molar_kcal_mol", target + "_molar_kcal_mol_ensemble_uncal_var"]
    for target in ["DeltaG_ET_gas", "DeltaG_EP_gas", "DeltaDeltaG_ET_solv", "DeltaDeltaG_EP_solv", "DeltaG_ET_soln", "DeltaG_EP_soln"]:
        new_columns += [target + f"_molar_kcal_mol_model_{i}" for i in range(n_models)]

    # Save the results
    df_pred[new_columns].to_csv(preds_path, index=False)
    os.remove(preds_path + ".gas")
    os.remove(preds_path + ".solv")


if __name__ == "__main__":
    main()
