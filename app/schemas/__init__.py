from __future__ import annotations

import importlib
import inspect
import pkgutil
import types as _types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, Field, create_model


def _strip_annotation(ann: Any, cache: dict) -> Any:
    """타입 어노테이션에서 Pydantic 모델을 NoDesc 버전으로 교체."""
    origin = get_origin(ann)
    args = get_args(ann)

    # X | Y (Python 3.10+ union) 또는 Optional[X] / Union[X, Y]
    if isinstance(ann, _types.UnionType) or origin is Union:
        new_args = tuple(_strip_annotation(a, cache) for a in args)
        result = new_args[0]
        for a in new_args[1:]:
            result = result | a
        return result

    # list[X]
    if origin is list:
        if args:
            return list[_strip_annotation(args[0], cache)]
        return list

    # BaseModel 하위 클래스 → 재귀적으로 description 제거
    if isinstance(ann, type) and issubclass(ann, BaseModel) and ann is not BaseModel:
        return strip_descriptions(ann, cache)

    return ann


def strip_descriptions(
    model_class: type[BaseModel], _cache: dict | None = None
) -> type[BaseModel]:
    """Pydantic 모델에서 Field description/title을 재귀적으로 제거한 복제본 생성."""
    if _cache is None:
        _cache = {}
    if model_class in _cache:
        return _cache[model_class]

    new_name = f"{model_class.__name__}NoDesc"
    # 순환 참조 방지용 placeholder
    _cache[model_class] = None  # type: ignore

    new_fields: dict[str, Any] = {}
    for name, field_info in model_class.model_fields.items():
        ann = field_info.annotation  # 해석된 타입 사용 (__annotations__는 문자열일 수 있음)
        stripped_ann = _strip_annotation(ann, _cache)

        if field_info.default_factory is not None:
            new_fields[name] = (stripped_ann, Field(default_factory=field_info.default_factory))
        elif field_info.is_required():
            new_fields[name] = (stripped_ann, ...)
        else:
            new_fields[name] = (stripped_ann, field_info.default)

    new_model = create_model(new_name, **new_fields)
    _cache[model_class] = new_model
    return new_model


_schema_registry: dict[str, type[BaseModel]] = {}


def _discover_schemas():
    package = importlib.import_module("app.schemas")
    for _importer, modname, _ispkg in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        module = importlib.import_module(modname)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseModel) and obj is not BaseModel:
                _schema_registry[name] = obj


_discover_schemas()

# NoDesc 변형을 자동 생성하여 등록
_no_desc_cache: dict[type[BaseModel], type[BaseModel]] = {}
for _name, _cls in list(_schema_registry.items()):
    _no_desc = strip_descriptions(_cls, _no_desc_cache)
    _schema_registry[_no_desc.__name__] = _no_desc


def get_schema(name: str) -> type[BaseModel]:
    if name not in _schema_registry:
        available = ", ".join(sorted(_schema_registry.keys()))
        raise KeyError(f"Schema '{name}' not found. Available: {available}")
    return _schema_registry[name]


def list_schemas() -> list[str]:
    return sorted(_schema_registry.keys())
