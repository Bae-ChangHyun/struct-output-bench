from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.models import ExtractionRequest, ExtractionResponse
from app.config import settings
from app.frameworks.registry import FrameworkRegistry
from app.prompts.loader import list_prompts, load_prompt
from app.schemas import get_schema, list_schemas

router = APIRouter(prefix="/api")


@router.get("/frameworks")
async def get_frameworks() -> dict:
    result = {}
    for name in FrameworkRegistry.list_names():
        result[name] = FrameworkRegistry.list_modes(name)
    return {"frameworks": result}


@router.get("/frameworks/{name}/modes")
async def get_framework_modes(name: str) -> dict:
    try:
        modes = FrameworkRegistry.list_modes(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"framework": name, "modes": modes}


@router.get("/schemas")
async def get_schemas() -> dict[str, list[str]]:
    return {"schemas": list_schemas()}


@router.get("/prompts")
async def get_prompts() -> dict[str, list[str]]:
    return {"prompts": list_prompts()}


@router.post("/extract", response_model=ExtractionResponse)
async def extract(req: ExtractionRequest) -> ExtractionResponse:
    try:
        adapter_cls = FrameworkRegistry.get(req.framework)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if req.mode not in adapter_cls.supported_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Mode '{req.mode}' not supported by '{req.framework}'. "
            f"Available: {adapter_cls.supported_modes}",
        )

    try:
        schema_class = get_schema(req.schema_name)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        prompt = load_prompt(req.prompt_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = req.model or settings.default_model
    base_url = req.base_url or settings.openai_base_url
    api_key = req.api_key.get_secret_value() if req.api_key else settings.openai_api_key.get_secret_value()

    adapter = adapter_cls(
        model=model, base_url=base_url, api_key=api_key, mode=req.mode,
    )
    result = await adapter.run(
        text=req.markdown,
        schema_class=schema_class,
        system_prompt=prompt.system_prompt,
    )
    return ExtractionResponse(**result.model_dump())
