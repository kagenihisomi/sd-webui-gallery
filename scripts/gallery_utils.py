import json
import logging
import re
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

from scripts.constants import PATH_OUTPUTS

SPACER = " | "

MAX_DISPLAY = 100


def filter_df_image_infos(
    evt: gr.SelectData,
    df_infos: pd.DataFrame,
    sub_folders,
    dates,
    models,
    prompts,
    max_display=MAX_DISPLAY,
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

    # Get random sample of max_display
    df_sampled = sample_df(df, max_display)

    return (
        df_sampled["path_full"].tolist(),
        df_sampled,
        gr.Dropdown.update(**calc_dropbox_updates(df, "sub_folder", sub_folders)),
        gr.Dropdown.update(**calc_dropbox_updates(df, "date", dates)),
        gr.Dropdown.update(**calc_dropbox_updates(df, "model", models)),
        gr.Dropdown.update(**calc_dropbox_updates(df, "prompt", prompts)),
    )


def sample_df(df: pd.DataFrame, max_display=MAX_DISPLAY):
    """
    Adding randomness to the gallery
    Shuffles the rows of a pandas DataFrame and returns a subset of the shuffled DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to shuffle.
    max_display (int): The maximum number of rows to return. If the DataFrame has less than max_display rows,
                       all rows will be returned.

    Returns:
    pd.DataFrame: A shuffled subset of the input DataFrame.
    """
    if len(df) < max_display:
        return df.sample(frac=1)
    else:
        return df.sample(max_display)


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


def parse_gen_info(s: str) -> dict:
    """
    Parses a string containing general information and returns a dictionary.
    e.g. 'Steps: 40, Sampler: DPM++ 2M Karras, CFG scale: 8, Seed: 1429876294, Size: 512x512, Model hash: 61bc7001e8, Model: anyhentai_20, Denoising strength: 0.7, Clip skip: 2, Hires upscale: 1.5, Hires steps: 10, Hires upscaler: Latent, Lora hashes: "concept_sunbathing: 7b93421e39bd", TI hashes: "EasyNegativeV2: 339cc9210f70, FastNegativeV2: a7465e7cc2a2, EasyNegativeV2: 339cc9210f70, FastNegativeV2: a7465e7cc2a2", Version: 1.6.0'

    The input string is expected to be in a specific format, with key-value pairs separated by commas.
    This function cleans up the string and converts it to a dictionary.

    Args:
        s (str): The input string to be parsed.

    Returns:
        dict: A dictionary containing the key-value pairs from the input string.
    """
    # Parsing Hashes in a PITA, remove
    # Split the string at the first occurrence of "Hashes" and clean
    parts = s.split("Hashes", 1)
    s = parts[0]

    # s = "a, b, c, , , " -> a,b,c
    s = re.sub(r"[,\s]+$", "", s)

    # Replace commas with quotes, but not in {}
    s = re.sub(r"(,\s)(?![^{]*})", '", "', s)
    s = re.sub(r"(:\s)(?![^{]*})", '": "', s)

    # Fix up { and } and ""
    s = re.sub('"{', "{", s)
    s = re.sub(': ""', ': {"', s)
    s = re.sub('"",', '"},', s)

    # Put quotes around the whole thing
    s = '"' + s

    # For the last one, we need to check if it's a } or not
    if s[-1] != "}":
        s = s + '"'

    s = "{" + s + "}"

    d = json.loads(s)
    return d


def parse_image_info_from_path(path: Path) -> dict:
    # Initialize dictionary with defaults
    d = {}
    d["path_full"] = path
    d["date"] = path.parent.stem
    d["sub_folder"] = path.parent.parent.stem
    d["model"] = ""
    d["steps"] = ""
    d["sampler"] = ""
    d["full_generation_info"] = {}
    d["prompt"] = []
    d["negative_prompt"] = []

    d["prompt_raw"] = ""
    d["negative_prompt_raw"] = ""

    # Handle extras folder, it doesn't have a date folder
    if "extras" in path.parent.stem:
        d["date"] = ""
        d["sub_folder"] = path.parent.stem

    image_info = Image.open(path).info
    if "parameters" not in image_info.keys():
        logging.info(f"{path}: No parameters")
        return d
    s = image_info["parameters"]
    if s == "None":
        logging.info(f"{path}: No parameters, possibly ONNX generated image")
        return d

    # Split off generation info
    steps_str = "Steps: "
    prompt_info, gen_info = s.split(steps_str)
    generation_info = parse_gen_info(steps_str + gen_info)
    d["model"] = generation_info["Model"]
    d["steps"] = generation_info["Steps"]
    d["sampler"] = generation_info["Sampler"]
    d["full_generation_info"] = generation_info

    # Parse prompt info
    if prompt_info:
        str_negative_prompt = "Negative prompt: "
        if str_negative_prompt in prompt_info:
            # has negative prompt
            prompts = prompt_info.split(str_negative_prompt)
            d["prompt"] = parse_prompt(prompts[0])
            d["prompt_raw"] = prompts[0]
            d["negative_prompt"] = parse_prompt(prompts[1])
            d["negative_prompt_raw"] = prompts[1]
        else:
            d["prompt"] = parse_prompt(prompt_info)
            d["prompt_raw"] = prompt_info

    return d


def calc_df_image_infos(path_outputs: Path = PATH_OUTPUTS):
    path_imgs = path_outputs.glob("**/*.png")
    image_infos = [parse_image_info_from_path(path) for path in path_imgs]
    df = pd.DataFrame.from_records(image_infos)
    return df


def calc_unique_options(df_infos: pd.DataFrame, column: str) -> list:
    value_counts = calc_counts(df_infos, column)
    # Create a list from the value counts like 1girl|3507 1boy|3507 1girl 1boy|3507

    l_counts = calc_dropbox_choices(value_counts)

    return l_counts


def calc_dropbox_choices(value_counts):
    l_counts = [
        f"{value}{SPACER}{count}"
        for value, count in zip(value_counts.index, value_counts.values)
    ]

    return l_counts


def calc_dropbox_updates(
    df_infos: pd.DataFrame, column: str, current_dropbox_values: list
):
    value_counts = calc_counts(df_infos, column)
    # Create a list from the value counts like 1girl|3507 1boy|3507 1girl 1boy|3507

    denumber_current_dropbox = denumber_list(current_dropbox_values)
    values = calc_dropbox_choices(value_counts.loc[denumber_current_dropbox])

    choices = calc_dropbox_choices(value_counts)

    return {"value": values, "choices": choices}


def calc_counts(df_infos, column):
    if column == "prompt":
        # prompt is a column of lists
        value_counts = df_infos[column].explode().value_counts()
    else:
        value_counts = df_infos[column].value_counts()
    return value_counts


def parse_prompt(prompt: str) -> list:
    """
    Parses a given prompt string and returns a list of cleaned prompt elements.

    Args:
        prompt (str): The prompt string to be parsed.

    Returns:
        list: A list of cleaned prompt elements.

    Example:
        detailed face, detailed expression, smile
        (long hair, pink hair, yellow hair:1.1)

        [detailed face, detailed expression, smile, long hair, pink hair, yellow hair]
    """
    # Remove (,),1.1
    prompt = re.sub(r"\d+\.\d+", "", prompt)
    prompt = re.sub(r"[()]", ",", prompt)

    # Deal with lora prompt
    prompt = re.sub(r"<", ",<", prompt)
    prompt = re.sub(r">", ">,", prompt)
    prompt = re.sub(":(?![^<]*>)", "", prompt)

    # Ensure newlines are treated as commas
    prompt = re.sub("\n", ",", prompt)
    prompt = re.sub(r"[,\s]{2,}", ",", prompt)

    # remove trailing whitespace, commas
    prompt = prompt.strip()
    prompt = prompt.strip(",")

    # return prompt
    l_prompt = prompt.split(",")
    return l_prompt


def load_img_path(path: Path) -> list:
    # Load image paths
    image_paths = [path_file for path_file in path.glob("**/*.png")]
    return image_paths


def load_img_info(path: Path) -> list:
    # Load image info
    image_infos = [
        parse_image_info_from_path(path_file) for path_file in path.glob("**/*.png")
    ]
    return image_infos


def denumber(s: str) -> str:
    return s.split(SPACER)[0]


def denumber_list(l: list) -> list:
    return [denumber(s) for s in l]
