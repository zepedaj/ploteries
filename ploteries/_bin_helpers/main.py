import climax as clx


@clx.group()
def main():
    pass


@clx.parent()
@clx.argument("path", help="Ploteries data store file path.")
def path_arg():
    pass
