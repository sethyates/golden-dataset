"""
Command-line interface for golden-dataset.
"""

import logging
import sys
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from golden_dataset.core import get_sqlalchemy_base, get_sqlalchemy_engine, get_sqlalchemy_session_factory, sum_dicts
from golden_dataset.exc import GoldenError
from golden_dataset.main import GoldenManager, GoldenSettings

app = typer.Typer(help="Golden dataset management")
console = Console()
settings = GoldenSettings()


@app.callback()
def main(
    datasets_dir: str = typer.Option(settings.datasets_dir, help="Directory containing datasets"),
    src_dir: str = typer.Option(settings.src_dir, help="Directory containing source"),
    loglevel: str | None = typer.Option("WARNING", help="Enable logging at the given log level"),
) -> None:
    """
    Generate and manage golden datasets.
    """
    settings.datasets_dir = datasets_dir
    settings.src_dir = src_dir

    if loglevel:
        logging.basicConfig(level=logging.getLevelName(loglevel))


@app.command("list")
def list_datasets() -> None:
    """
    List all available golden datasets.
    """
    try:
        manager = GoldenManager(settings=settings)
        datasets = manager.list_datasets()

        if not datasets:
            console.print("[yellow]No datasets found[/yellow]")
            return

        # Create a table with dataset information
        table = Table(title="Golden Datasets")
        table.add_column("Name", style="cyan")
        table.add_column("Title", style="cyan")
        table.add_column("Tables", style="magenta", justify="right")
        table.add_column("Records", style="blue", justify="right")
        table.add_column("Dependencies", style="green")
        table.add_column("Exported At", style="yellow")

        datasets = sorted(datasets, key=lambda x: x.name)

        for metadata in datasets:
            total_records = sum(metadata.tables.values())
            table_count = len(metadata.tables)

            table.add_row(
                metadata.name,
                metadata.title,
                str(table_count),
                str(total_records),
                ", ".join(metadata.dependencies),
                metadata.exported_at.isoformat(),
            )

        console.print(table)

    except GoldenError as e:
        raise typer.Exit(code=1) from e

    except FileNotFoundError as e:
        raise typer.Exit(code=1) from e

    except Exception as e:
        console.print(f"[red]Error listing datasets: {str(e)}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(code=1) from e


@app.command("show")
def show_dataset(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to import"),
) -> None:
    """
    Show details about a dataset.
    """
    try:
        manager = GoldenManager(settings=settings)
        dataset = manager.open_dataset(dataset_name)

        table = Table(title="Golden Dataset")
        table.add_section()
        table.add_column("Name", style="cyan")
        table.add_column("Title", style="cyan")
        table.add_column("Revision", style="blue")
        table.add_column("Description", style="white")
        table.add_column("Dependencies", style="green")
        table.add_column("Exported At", style="yellow")
        table.add_row(
            dataset.name,
            dataset.title,
            dataset.revision,
            dataset.description,
            ", ".join(dataset.dependencies),
            dataset.exported_at.isoformat(),
        )
        console.print(table)

        table = Table(title="Tables")
        table.add_column("Table", style="cyan")
        table.add_column("Records", style="magenta", justify="right")
        for name, records in dict(sorted(dataset.tables.items())).items():
            table.add_row(name, str(records))
        console.print(table)

    except GoldenError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    except Exception as e:
        console.print(f"[red]Error showing dataset: {str(e)}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(code=1) from e


@app.command("generate")
def generate_dataset(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to generate"),
    generators: str = typer.Option(settings.generators, help="Module containing generators"),
    base_class_name: str = typer.Option(settings.base_class_name, help="Base class name"),
    engine_name: str = typer.Option(settings.engine_name, help="Engine instance name"),
    session_factory_name: str = typer.Option(settings.session_factory_name, help="Session factory name"),
) -> None:
    """
    Generate a golden dataset from a generator function.

    The generator function should be decorated with @golden and take a session as argument.
    """
    try:
        console.rule()
        console.print(f"Generating dataset from [bold]{dataset_name}[/bold]...")

        settings.generators = generators
        settings.base_class_name = base_class_name
        settings.engine_name = engine_name
        settings.session_factory_name = session_factory_name

        # Generate the dataset
        manager = GoldenManager(settings=settings)
        dataset = manager.generate_dataset(dataset_name)
        manager.dump_dataset(dataset)

        console.print("[green]Dataset generated successfully![/green]")
        console.rule()

        # Show table information
        table = Table(title=f"Tables in {dataset.name}")
        table.add_column("Table", style="cyan")
        table.add_column("Records", style="magenta")

        for table_name, count in dataset.get_tables().items():
            table.add_row(table_name, str(count))

        console.print(table)

    except GoldenError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    except Exception as e:
        console.print(f"[red]Error generating dataset: {str(e)}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(code=1) from e


def recursively_load_datasets(
    dataset_name: str, base: Any, session: Any, recurse: bool = True, marks: dict[str, bool] | None = None
) -> dict[str, int]:
    if marks is None:
        marks = dict[str, bool]()

    if dataset_name in marks:
        return {}

    marks[dataset_name] = True

    results: dict[str, int] = dict()
    manager = GoldenManager(settings=settings)
    dataset = manager.load_dataset(dataset_name)

    if dataset is not None and recurse:
        for dependency in dataset.dependencies:
            results = sum_dicts(results, recursively_load_datasets(dependency, base, session, marks=marks))

    console.print(f"[green]Loading {dataset_name}[/green]")
    results = sum_dicts(results, dataset.add_to_session(base, session))
    return results


@app.command("load")
def load_dataset(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to load"),
    depends: bool = typer.Option(True, help="Whether to load dependencies or not"),
) -> None:
    """
    Load a dataset into a database.
    """
    try:
        sys.path.insert(0, "")

        engine = get_sqlalchemy_engine(
            engine_name=settings.engine_name,
            package=settings.src_dir,
        )
        if not engine:
            console.print("[red]Could not find engine[/red]")
            return

        base = get_sqlalchemy_base(
            base_class_name=settings.base_class_name,
            package=settings.src_dir,
        )
        if not base:
            console.print("[red]Could not find Base Base[/red]")
            return

        sessionmaker = get_sqlalchemy_session_factory(
            session_factory_name=settings.session_factory_name,
            package=settings.src_dir,
        )
        if not sessionmaker:
            console.print("[red]Could not find Session[/red]")
            return

        base.metadata.create_all(bind=engine)

        with sessionmaker() as session:
            try:
                results = recursively_load_datasets(dataset_name, base, session, recurse=depends)
                session.commit()
            except Exception as e:
                session.rollback()
                raise e

        table = Table(title="Results")
        table.add_column("Table", style="cyan")
        table.add_column("Records", style="cyan", justify="right")

        for name, records in results.items():
            table.add_row(name, str(records))

        console.print(table)

        console.print(f"[green]Dataset {dataset_name} imported successfully![/green]")

    except GoldenError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    except Exception as e:
        console.print(f"[red]Error loading dataset: {str(e)}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(code=1) from e


@app.command("unload")
def unload_dataset(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to unload"),
) -> None:
    """
    Remove a dataset into a database.
    """
    try:
        sys.path.insert(0, "")

        engine = get_sqlalchemy_engine(
            engine_name=settings.engine_name,
            package=settings.src_dir,
        )
        if not engine:
            console.print("[red]Could not find engine[/red]")
            return

        base = get_sqlalchemy_base(
            base_class_name=settings.base_class_name,
            package=settings.src_dir,
        )
        if not base:
            console.print("[red]Could not find Base Base[/red]")
            return

        sessionmaker = get_sqlalchemy_session_factory(
            session_factory_name=settings.session_factory_name,
            package=settings.src_dir,
        )
        if not sessionmaker:
            console.print("[red]Could not find Session[/red]")
            return

        with sessionmaker() as session:
            try:
                manager = GoldenManager(settings=settings)
                dataset = manager.load_dataset(dataset_name)
                results = dataset.remove_from_session(base, session)
                session.commit()
            except Exception as e:
                session.rollback()
                raise e

        table = Table(title="Results")
        table.add_column("Table", style="cyan")
        table.add_column("Records", style="cyan", justify="right")

        for name, records in results.items():
            table.add_row(name, str(records))

        console.print(table)

        console.print(f"[green]Dataset {dataset_name} removed successfully![/green]")

    except GoldenError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    except Exception as e:
        console.print(f"[red]Error unloading dataset: {str(e)}[/red]")
        import traceback

        console.print(traceback.format_exc())
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
