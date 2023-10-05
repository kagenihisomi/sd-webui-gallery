from pathlib import Path

PATH_PACKAGE = Path(__file__).parent.parent

PATH_OUTPUTS = PATH_PACKAGE / Path("../../outputs")
paths_folders = [i for i in PATH_OUTPUTS.glob("*") if i.is_dir()]
