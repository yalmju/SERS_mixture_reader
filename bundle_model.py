"""Package weights and raw gate templates once; prediction then needs no training maps."""
import argparse

import algorithm as A
import config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    model = config.load_reader_model(args.model)
    templates = A.class_templates(config.templates())
    provenance = dict(model.get("provenance", {}), source_model_sha256=config.file_hash(args.model),
                      template_sources={name: {p.replace(config.data_root(), "<data>"): config.file_hash(p)
                                               for p in paths} for name, paths in config.templates().items()})
    A.save_model(args.output, model, templates=templates, provenance=provenance)
    print("saved standalone model:", args.output)


if __name__ == "__main__":
    main()
