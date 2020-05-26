from datetime import datetime
from os.path import join
import pandas as pd

report_folder = "reports/"

translations = {
    "prob_name": "Problem",
    "prob_ansr": "Truth",
    "optn": "Prediction",
    "prob_type": "Problem Type",
    "anlg_name": "Analogy",
    "anlg_type": "Analogy Type",
    "tran_name": "Transformation",
    "tran_type": "Transformation Type",
    "mat_score": "MAT Score",
    "optn_score": "O Score",
    "mato_score": "MATO Score",
    "anlg_n": "# of Hits",
    "tran_n": "# of Hits",
    "crct_probs": "Correctly Answered",
    "incr_probs": "Incorrectly Answered",
    "crct/incr": "Correct v.s. Incorrect"
}

sltn_cols = ["prob_name", "prob_type", "prob_ansr", "optn",
             "anlg_name", "anlg_type", "tran_name", "tran_type", "mat_score", "optn_score", "mato_score"]
sltn_hdrs = [translations.get(col) for col in sltn_cols]
sltn_col_widths = [15, 15, 15, 15, 40, 15, 20, 15, 15, 15, 15]

sltn_anlg_cols = ["anlg_name", "anlg_type", "anlg_n", "crct_probs", "incr_probs", "crct/incr"]
sltn_anlg_hdrs = [translations.get(col) for col in sltn_anlg_cols]
sltn_anlg_col_widths = [40, 15, 15, 40, 15, 15]

sltn_tran_cols = ["tran_name", "tran_type", "tran_n", "crct_probs", "incr_probs", "crct/incr"]
sltn_tran_hdrs = [translations.get(col) for col in sltn_tran_cols]
sltn_tran_col_widths = [25, 15, 15, 70, 20, 15]


def create_report(probs, prefix):

    file_name = prefix + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".xlsx"

    _, _, _, sltn_df, sltn_anlg_df, sltn_tran_df = extract_data(probs)

    with pd.ExcelWriter(join(report_folder, file_name)) as writer:

        create_sheet(writer, "Problems", sltn_df, sltn_col_widths, sltn_hdrs)
        create_sheet(writer, "Analogies", sltn_anlg_df, sltn_anlg_col_widths, sltn_anlg_hdrs)
        create_sheet(writer, "Transformations", sltn_tran_df, sltn_tran_col_widths, sltn_tran_hdrs)

        highlight(writer, "Problems", sltn_df)


def highlight(writer, sheet_name, df):
    red = writer.book.add_format({"color": "#FF0000"})
    bold = writer.book.add_format({'bold': True, 'text_wrap': True})
    sheet = writer.sheets[sheet_name]
    correct_n = 0
    for ii in range(len(df)) :
        row = df.iloc[ii]
        if row["prob_ansr"] != row["optn"]:
            sheet.set_row(ii + 1, None, red)
        else:
            correct_n += 1

    sheet.write(ii + 2, 0, "Accuracy:", bold)
    sheet.write(ii + 2, 1, str(correct_n) + "/" + str(ii + 1), bold)


def create_sheet(writer, sheet_name, df, col_widths, col_hdrs):
    df.to_excel(writer, sheet_name = sheet_name, index_label = col_hdrs[0], header = col_hdrs[1:])
    sheet = writer.sheets[sheet_name]
    for jj, w in enumerate(col_widths):
        sheet.set_column(jj, jj, w)


def extract_data(probs):

    global sltn_cols

    anlg_tran_data = []
    anlg_data = []
    pred_data = []
    sltn_data = []
    for prob in probs:
        aggregation_progression = prob.data
        anlg_tran_data.extend(aggregation_progression.get("anlg_tran_data"))
        anlg_data.extend(aggregation_progression.get("anlg_data", []))
        pred_data.extend(aggregation_progression.get("pred_data"))
        sltn_data.append(aggregation_progression.get("pred_d"))

    anlg_tran_df = pd.DataFrame(data = anlg_tran_data)
    anlg_df = pd.DataFrame(data = anlg_data)
    pred_df = pd.DataFrame(data = pred_data)
    sltn_df = pd.DataFrame(data = sltn_data, columns = sltn_cols)

    sltn_anlg_df = sltn_df.groupby("anlg_name").apply(sltn2anlg)
    sltn_tran_df = sltn_df.groupby("tran_name").apply(sltn2tran)
    sltn_df.set_index("prob_name", inplace = True)

    return anlg_tran_df, anlg_df, pred_df, sltn_df, sltn_anlg_df, sltn_tran_df


def sltn2anlg(anlg_group):
    correct_ones = anlg_group.loc[anlg_group["prob_ansr"] == anlg_group["optn"]]
    incorrect_ones = anlg_group.loc[anlg_group["prob_ansr"] != anlg_group["optn"]]
    d = {
        "anlg_type": anlg_group["anlg_type"].iloc[0],
        "anlg_n": len(anlg_group),  # row number
        "crct_probs": ','.join(correct_ones["prob_name"].to_list()),
        "incr_probs": ','.join(incorrect_ones["prob_name"].to_list()),
        "crct/incr": str(len(correct_ones)) + '/' + str(len(incorrect_ones))
    }
    return pd.Series(d)


def sltn2tran(tran_group):
    correct_ones = tran_group.loc[tran_group["prob_ansr"] == tran_group["optn"]]
    incorrect_ones = tran_group.loc[tran_group["prob_ansr"] != tran_group["optn"]]
    d = {
        "tran_type": tran_group["tran_type"].iloc[0],
        "tran_n": len(tran_group),
        "crct_probs": ','.join(correct_ones["prob_name"].to_list()),
        "incr_probs": ','.join(incorrect_ones["prob_name"].to_list()),
        "crct/incr": str(len(correct_ones)) + '/' + str(len(incorrect_ones))
    }
    return pd.Series(d)