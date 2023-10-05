from functools import partial
import gradio as gr
import pandas as pd

from sd_webui_gallery.images import (
    calc_counts,
    calc_df_image_infos,
    calc_dropbox_updates,
    calc_unique_options,
)
from sd_webui_gallery.images import denumber_list

DROPDOWN_SELECT = partial(
    gr.Dropdown, interactive=True, multiselect=True, show_label=True
)


def main():
    with gr.Blocks() as demo:
        # UI Components
        df_image_infos = calc_df_image_infos()
        prompts = calc_unique_options(df_image_infos, "prompt")
        models = calc_unique_options(df_image_infos, "model")
        dates = calc_unique_options(df_image_infos, "date")
        sub_folder = calc_unique_options(df_image_infos, "sub_folder")
        grdf_image_info = gr.DataFrame(df_image_infos, visible=False)
        grdf_image_info_filtered = gr.DataFrame(df_image_infos, visible=False)

        with gr.Row():
            with gr.Column():
                image_gallery = gr.Gallery(
                    value=df_image_infos["path_full"].tolist(),
                    label="Select an image to view information",
                    show_label=True,
                    columns=3,
                    preview=False,
                )

            with gr.Column():
                gr.Textbox("Search and filter images", show_label=False)

                sub_folder_dropdown = DROPDOWN_SELECT(
                    value=[],
                    choices=sub_folder,
                    label="Folder",
                )
                date_dropdown = DROPDOWN_SELECT(
                    value=[],
                    choices=dates,
                    label="Date",
                )
                model_dropdown = DROPDOWN_SELECT(
                    value=[],
                    choices=models,
                    label="Model",
                )
                prompts_dropdown = DROPDOWN_SELECT(
                    value=[],
                    choices=prompts,
                    label="Prompts",
                )

                textbox_image_info = gr.Textbox("", show_label=False)

        # Drop UI Callbacks
        dropdowns = [
            sub_folder_dropdown,
            date_dropdown,
            model_dropdown,
            prompts_dropdown,
        ]

        for dropdown in dropdowns:
            dropdown.select(
                filter_df_image_infos,
                [grdf_image_info, *dropdowns],
                [image_gallery, grdf_image_info_filtered, *dropdowns],
            )

        image_gallery.select(
            display_image_info, [grdf_image_info_filtered], textbox_image_info
        )

    demo.launch(inbrowser=True)


def filter_df_image_infos(
    evt: gr.SelectData, df_infos: pd.DataFrame, sub_folders, dates, models, prompts
):
    df = df_infos.copy()

    sub_folders_denumbered = denumber_list(sub_folders)
    dates_denumbered = denumber_list(dates)
    models_denumbered = denumber_list(models)
    prompts_denumbered = denumber_list(prompts)

    filters = {
        "sub_folder": sub_folders_denumbered,
        "date": dates_denumbered,
        "model": models_denumbered,
        "prompt": prompts_denumbered,
    }

    for column, values in filters.items():
        if not values:
            # No filter means all values are selected
            continue
        if column == "prompt":
            # prompt is a column of lists
            df = df[df[column].apply(lambda x: set(values).issubset(set(x)))]
        else:
            df = df[df[column].isin(values)]

    path_full_list = df["path_full"].tolist()

    return (
        path_full_list,
        df,
        gr.Dropdown(**calc_dropbox_updates(df, "sub_folder", sub_folders)),
        gr.Dropdown(**calc_dropbox_updates(df, "date", dates)),
        gr.Dropdown(**calc_dropbox_updates(df, "model", models)),
        gr.Dropdown(**calc_dropbox_updates(df, "prompt", prompts)),
    )


text_info = """
Prompt
{prompt}
Negative Prompt
{negative_prompt}
Generation Data
{generation_data}
"""


def display_image_info(evt: gr.SelectData, df_info: pd.DataFrame):
    df_select = df_info.iloc[evt.index, :]

    txt = text_info.format(
        prompt=df_select["prompt_raw"],
        negative_prompt=df_select["negative_prompt_raw"],
        generation_data=df_select["full_generation_info"],
    )

    return txt


if __name__ == "__main__":
    main()
