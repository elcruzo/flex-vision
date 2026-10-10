"""Strict data-only YAML frontend for the implemented detector subset."""
from pathlib import Path
import yaml
from .pipeline import Pipeline


class ConfigError(ValueError):
    """A configuration cannot describe a supported graph."""


class _Loader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ConfigError(f"line {key_node.start_mark.line+1}: mapping keys must be strings")
        if key in result:
            raise ConfigError(f"line {key_node.start_mark.line+1}: duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _fields(value, required, optional, location):
    if not isinstance(value, dict):
        raise ConfigError(f"{location}: expected a mapping")
    unknown = value.keys() - required - optional
    missing = required - value.keys()
    if unknown:
        raise ConfigError(f"{location}: unknown fields {sorted(unknown)}")
    if missing:
        raise ConfigError(f"{location}: missing fields {sorted(missing)}")
    return value


def load_pipeline(path):
    """Read a version-1 detector configuration. No execution or GPU allocation."""
    path = Path(path)
    try:
        with path.open('rb') as source:
            raw = source.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ConfigError("configuration exceeds 1 MiB")
        data = yaml.load(raw, Loader=_Loader)
        _fields(data, {"input", "pipeline", "output"}, {"schema_version"}, "root")
        version = data.get("schema_version", 1)
        if type(version) is not int or version != 1:
            raise ConfigError("schema_version: only version 1 is supported")
        inp = _fields(data["input"], {"encoding"}, set(), "input")
        try:
            pipeline = Pipeline(input_encoding=inp["encoding"])
        except ValueError as exc:
            raise ConfigError(f"input.encoding: {exc}") from exc
        operations = data["pipeline"]
        if not isinstance(operations, list) or len(operations) not in (2, 3):
            raise ConfigError("pipeline: require optional convolution, letterbox, then normalize")
        offset = 0
        if len(operations) == 3:
            _fields(operations[0], {'convolution'}, set(), 'pipeline[0]')
            parameters = _fields(operations[0]['convolution'], {'kernel'}, set(), 'pipeline[0].convolution')
            kernel = parameters['kernel']
            if isinstance(kernel, str):
                kernel_path = path.parent / kernel
                try:
                    with kernel_path.open('rb') as source: raw_kernel = source.read(65_537)
                    if len(raw_kernel) > 65_536: raise ConfigError('kernel file exceeds 64 KiB')
                    kernel_data = yaml.load(raw_kernel, Loader=_Loader)
                    kernel = _fields(kernel_data, {'kernel'}, set(), 'kernel file')['kernel']
                except (OSError, yaml.YAMLError, ConfigError, UnicodeError) as exc:
                    raise ConfigError(f'pipeline[0].convolution.kernel {kernel_path}: {exc}') from exc
            try:
                pipeline = pipeline.conv2d(kernel)
            except (ValueError, TypeError) as exc:
                raise ConfigError(f'pipeline[0].convolution: {exc}') from exc
            offset = 1
        for i, (name, required, optional) in enumerate([
            ("letterbox", {"width", "height"}, {"value"}),
            ("normalize", {"mean", "std", "scale"}, set()),
        ]):
            i += offset
            location = f"pipeline[{i}].{name}"
            _fields(operations[i], {name}, set(), f"pipeline[{i}]")
            parameters = _fields(operations[i][name], required, optional, location)
            try:
                pipeline = getattr(pipeline, name)(**parameters)
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"{location}: {exc}") from exc
        output = _fields(data["output"], {"dtype", "layout"}, set(), "output")
        if not isinstance(output["layout"], str):
            raise ConfigError("output.layout: expected NCHW")
        try:
            return pipeline.to(dtype=output["dtype"], layout=output["layout"].upper())
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"output: {exc}") from exc
    except (OSError, yaml.YAMLError, ConfigError, UnicodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
