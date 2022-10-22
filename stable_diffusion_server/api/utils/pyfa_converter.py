import inspect
from typing import Callable, Any, Type, Union, List

import typing
from fastapi import Depends, Form, Query, Body
from pydantic import BaseModel
from pydantic.fields import ModelField, FieldInfo


# plucked from dotX12/pyfa-converter (MIT), for use until tiangolo/fastapi#4700 is fixed


class PydanticConverterUtils:
    @classmethod
    def param_maker(
        cls, field: ModelField, _type: Callable[..., FieldInfo]
    ) -> FieldInfo:
        if field.required is True:
            return cls.__fill_params(param=_type, model_field=field, default=...)
        return cls.__fill_params(param=_type, model_field=field, default=field.default)

    @classmethod
    def __fill_params(
        cls, param: Callable[..., FieldInfo], model_field: ModelField, default: Any
    ) -> FieldInfo:
        return param(
            default=default or None,
            alias=model_field.field_info.alias or None,
            title=model_field.field_info.title or None,
            description=model_field.field_info.description or None,
            gt=model_field.field_info.gt or None,
            ge=model_field.field_info.ge or None,
            lt=model_field.field_info.lt or None,
            le=model_field.field_info.le or None,
            min_length=model_field.field_info.min_length or None,
            max_length=model_field.field_info.max_length or None,
            regex=model_field.field_info.regex or None,
            **model_field.field_info.extra,
        )

    @classmethod
    def override_signature_parameters(
        cls,
        model: Type[BaseModel],
        param_maker: Callable[[ModelField], Any],
    ) -> List[inspect.Parameter]:
        return [
            inspect.Parameter(
                name=field.alias,
                kind=inspect.Parameter.POSITIONAL_ONLY,
                default=param_maker(field),
                annotation=field.outer_type_,
            )
            for field in model.__fields__.values()
        ]


class PydanticConverter(PydanticConverterUtils):
    @classmethod
    def reformat_model_signature(
        cls,
        model_cls: Type[BaseModel],
        _type: Any,
    ) -> Union[Type[BaseModel], Type["PydanticConverter"]]:
        """
        Adds an `query` class method to decorated models. The `query` class
        method can be used with `FastAPI` endpoints.
        Args:
            model_cls: The model class to decorate.
            _type: literal query, form, body, etc...
        Returns:
            The decorated class.
        """
        _type_var_name = str(_type.__name__).lower()

        def make_form_parameter(field: ModelField) -> Any:
            """
            Converts a field from a `Pydantic` model to the appropriate `FastAPI`
            parameter type.
            Args:
                field: The field to convert.
            Returns:
                Either the result of `Query`, if the field is not a sub-model, or
                the result of `Depends on` if it is.
            """
            if type(field.type_) is type and issubclass(field.type_, BaseModel):
                # This is a sub-model.
                assert hasattr(field.type_, _type_var_name), (
                    f"Sub-model class for {field.name} field must be decorated with"
                    f" `as_form` too."
                )
                attr = getattr(field.type_, _type_var_name)
                return Depends(attr)  # noqa
            else:
                return cls.param_maker(field=field, _type=_type)

        new_params = PydanticConverterUtils.override_signature_parameters(
            model=model_cls, param_maker=make_form_parameter
        )

        def _as_form(**data) -> Union[BaseModel, PydanticConverter]:
            return model_cls(**data)

        sig = inspect.signature(_as_form)
        sig = sig.replace(parameters=new_params)
        _as_form.__signature__ = sig
        setattr(model_cls, _type_var_name, _as_form)
        return model_cls


class PyFaDepends:
    def __new__(
        cls,
        model: Type[BaseModel],
        _type: Callable[..., FieldInfo],
    ):
        return cls.generate(model=model, _type=_type)

    @classmethod
    def generate(
        cls,
        model: Type[BaseModel],
        _type: Callable[..., FieldInfo],
    ):
        obj = PydanticConverter.reformat_model_signature(model_cls=model, _type=_type)
        attr = getattr(obj, str(_type.__name__).lower())
        return Depends(attr)


class BodyDepends(PyFaDepends):
    _TYPE = Body

    def __new__(cls, model_type: Type[BaseModel]):
        return super().generate(model=model_type, _type=cls._TYPE)


class FormDepends(PyFaDepends):
    _TYPE = Form

    def __new__(cls, model_type: Type[BaseModel]):
        return super().generate(model=model_type, _type=cls._TYPE)


class QueryDepends(PyFaDepends):
    _TYPE = Query

    def __new__(cls, model_type: Type[BaseModel]):
        return super().generate(model=model_type, _type=cls._TYPE)
