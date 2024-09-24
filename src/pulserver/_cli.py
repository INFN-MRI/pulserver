"""Command Line Interface for server."""

import os

import click

from . import server


@click.group()
def cli():
    pass


@click.command()
@click.option("--config-path", default=None, help="Location of config.yml file.")
@click.option("--scanner-address", default=None, help="IP address of MR scanner.")
@click.option("--scanner-port", default=None, help="MR scanner port.")
@click.option(
    "--recon-server-address", default=None, help="IP address of reconstruction servr."
)
@click.option("--recon-server-port", default=None, help="Reconstruction server port.")
def start(
    config_path, scanner_address, scanner_port, recon_server_address, recon_server_port
):
    """
    Start sequence design server.
    """
    if config_path is not None:
        os.environ["PULSEFORGE_CONFIG"] = config_path
    else:
        if scanner_address is not None:
            os.environ["PULSEFORGE_SCANNER_ADDRESS"] = scanner_address
        if scanner_port is not None:
            os.environ["PULSEFORGE_SCANNER_PORT"] = scanner_port
        if recon_server_address is not None:
            os.environ["PULSEFORGE_RECON_SERVER_ADDRESS"] = recon_server_address
        if recon_server_port is not None:
            os.environ["PULSEFORGE_RECON_SERVER_PORT"] = recon_server_port

    # start the server
    server.start_server()


@click.command()
@click.option(
    "--name",
    default=None,
    help="Function whose docstring we want to see. If not provided, print list of available commands.",
)
def apps(name):
    """
    Design function documentations
    """
    # Load plugin list
    plugins = server.load_plugins()
    if name is None:
        for fun in plugins.values():
            name = fun.__name__
            summary = fun.__doc__.split("\n")[1]
            click.echo(name + "\t" + summary)
    else:
        click.echo(plugins[name].__doc__)


cli.add_command(start)
cli.add_command(apps)

if __name__ == "__main__":
    cli()