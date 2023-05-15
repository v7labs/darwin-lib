from typing import Dict

from darwin.future.data_objects import validators as darwin_validators


class TeamSlug(str):
    """Team slug type"""

    min_length = 1
    max_length = 256

    @classmethod
    def __get_validators__(cls): #type: ignore
        yield cls.validate

    @classmethod
    def validate(cls, v: str) -> "TeamSlug":
        assert len(v) < cls.max_length, f'maximum length for team slug is {cls.max_length}'
        assert len(v) > cls.min_length, f'minimum length for team slug is {cls.min_length}'
        if not isinstance(v, str):
            raise TypeError('string required')
        modified_value = darwin_validators.parse_name(v)
        return cls(modified_value)
    
    def __repr__(self) -> str:
        return f'TeamSlug({super().__repr__()})'


class QueryString:
    """Query string type"""

    def __init__(self, value: Dict[str, str]) -> None:
        assert isinstance(value, dict)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())

        self.value = value

    def __str__(self) -> str:
        return "?" + "&".join(f"{k}={v}" for k, v in self.value.items())
