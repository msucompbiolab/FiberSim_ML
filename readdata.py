import pandas as pd


def read_excel_data(file_path: str = "parameter_values.xlsx") -> pd.DataFrame:
    """Read an Excel spreadsheet into a pandas DataFrame."""
    return pd.read_excel(file_path)


def read_tab_delimited_data(file_path: str = "summary_n_vars_1_part_1.txt") -> pd.DataFrame:
    """Read a tab-delimited text file into a pandas DataFrame."""
    return pd.read_csv(file_path, sep="\t")


if __name__ == "__main__":
    excel_df = read_excel_data("parameter_values.xlsx")
    tab_df = read_tab_delimited_data("summary_n_vars_1_part_1.txt")

    print("Excel data:")
    print(excel_df.head())
    print(excel_df.shape)

    print("\nTab-delimited data:")
    print(tab_df.head())
    print(tab_df.shape)
