"""
@title

@description

"""
import argparse
from pathlib import Path

from pyboard import project_properties


class LLM:

    @property
    def model_path(self):
        return Path(self.base_dir, f'{self.name}.mdl')

    def __init__(self, model_arch: str, model_id, tag='', base_dir=None):
        self.name = model_arch
        self.model_id = model_id
        self.tag = tag

        model_arch = model_arch.replace('/', '_')
        model_arch = model_arch.replace('\\', '_')

        self.name = f'{model_arch}_{model_id}_{tag}'
        self.base_dir = Path(project_properties.model_dir) if base_dir is None else Path(base_dir)
        return

    def respond(self, query):
        return

    def load(self):
        return

    def save(self, base_dir=None, name=None):
        if base_dir is None:
            base_dir = self.base_dir
        if name is None:
            name = self.name


        return


def main(main_args):
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
