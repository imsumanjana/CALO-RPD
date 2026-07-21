"""Export verified results for one experiment."""

import argparse
from calo_rpd_studio.results.database import ResultDatabase
from calo_rpd_studio.results.publication_export import PublicationExporter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--database", default="calo_rpd_results.sqlite")
    p.add_argument("--experiment", required=True)
    p.add_argument("--output", default="publication_export")
    a = p.parse_args()
    path = PublicationExporter(ResultDatabase(a.database)).export(a.experiment, a.output)
    print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
