from functools import partial

import gradio as gr

# pylint: disable=import-error, undefined-variable
from modules import script_callbacks

from scripts.utils import (
    calc_df_image_infos,
    calc_unique_options,
    display_image_info,
    filter_df_image_infos,
)

DROPDOWN_SELECT = partial(
    gr.Dropdown, interactive=True, multiselect=True, show_label=True
)


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as gallery:
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
                    value=df_image_infos["path_full"].tolist()[:9],
                    label="Select an image to view information",
                    show_label=True,
                    columns=3,
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

    return ((gallery, "Image Gallery", "gallery"),)


script_callbacks.on_ui_tabs(on_ui_tabs)
