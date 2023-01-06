from utils.general import colorstr


def print_debug_msg(message: str) -> None:
    print(colorstr("yellow", "bold", "DEBUGGING: ") + colorstr("yellow", message))
