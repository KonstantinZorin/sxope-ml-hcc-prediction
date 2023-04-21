def lowest_level(tpl: tuple) -> tuple:
    if isinstance(tpl[-1], tuple):
        return lowest_level(tpl[-1])
    else:
        return tpl
