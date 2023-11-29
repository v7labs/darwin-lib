from __future__ import annotations

from abc import ABC
from datetime import datetime
from typing import Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, validator

T = TypeVar("T")

AcceptedFileTypes = Literal["image", "video", "pdf", "dicom"]
IssueType = Literal["comment"]
ProcessingStatusType = Literal["cancelled", "error", "uploading", "uploading_confirmed", "processing", "complete"]
WorkflowStatusType = Literal["new", "annotate", "review", "complete"]


class BaseGroupFilter(BaseModel, ABC):
    conjuction: Literal['and', 'or'] = 'and'
    filters: List[BaseGroupFilter | BaseSubjectFilter]


class BaseSubjectFilter(BaseModel, ABC):
    subject: str
    matcher: BaseMatcher
    
class BaseMatcher(BaseModel, ABC):
    name: str
    
# Subject Filters
class AnnotationClass(BaseSubjectFilter):
    subject: Literal['annotation_class'] = 'annotation_class'
    matcher: AnyOf[int] | AllOf[int] | NoneOf[int]
    
    @classmethod
    def any_of(cls, values: list[int]) -> AnnotationClass:
        return AnnotationClass(subject='annotation_class', matcher=AnyOf(values=values))
    @classmethod
    def all_of(cls, values: list[int]) -> AnnotationClass:
        return AnnotationClass(subject='annotation_class', matcher=AllOf(values=values))
    @classmethod
    def none_of(cls, values: list[int]) -> AnnotationClass:
        return AnnotationClass(subject='annotation_class', matcher=NoneOf(values=values))

class Archived(BaseSubjectFilter):
    subject: Literal['archived'] = 'archived'
    matcher: Equals[bool]
    
    @classmethod
    def equals(cls, value: bool) -> Archived:
        return Archived(subject='archived', matcher=Equals(value=value))

class Assignee(BaseSubjectFilter):
    subject: Literal['assignee'] = 'assignee'
    matcher: AnyOf[int] | AllOf[int] | NoneOf[int]
    
    @classmethod
    def any_of(cls, values: list[int]) -> Assignee:
        return Assignee(subject='assignee', matcher=AnyOf(values=values))
    @classmethod
    def all_of(cls, values: list[int]) -> Assignee:
        return Assignee(subject='assignee', matcher=AllOf(values=values))
    @classmethod
    def none_of(cls, values: list[int]) -> Assignee:
        return Assignee(subject='assignee', matcher=NoneOf(values=values))
    
    
class CreatedAt(BaseSubjectFilter):
    subject: Literal['created_at'] = 'created_at'
    matcher: DateRange
    
    @classmethod
    def between(cls, start: datetime, end: datetime) -> CreatedAt:
        return CreatedAt(subject='created_at', matcher=DateRange(start=start, end=end))
    
    @classmethod
    def before(cls, end: datetime) -> CreatedAt:
        return CreatedAt(subject='created_at', matcher=DateRange(end=end))
    
    @classmethod
    def after(cls, start: datetime) -> CreatedAt:
        return CreatedAt(subject='created_at', matcher=DateRange(start=start))
    
    
class CurrentAssignee(BaseSubjectFilter):
    subject: Literal['current_assignee'] = 'current_assignee'
    matcher: AnyOf[int] | NoneOf[int]
    
    @classmethod
    def any_of(cls, values: list[int]) -> CurrentAssignee:
        return CurrentAssignee(subject='current_assignee', matcher=AnyOf(values=values))
    
    @classmethod
    def none_of(cls, values: list[int]) -> CurrentAssignee:
        return CurrentAssignee(subject='current_assignee', matcher=NoneOf(values=values))

class FileType(BaseSubjectFilter):
    subject: Literal['file_type'] = 'file_type'
    matcher: AnyOf[AcceptedFileTypes] | AllOf[AcceptedFileTypes] | NoneOf[AcceptedFileTypes]
    
    @classmethod
    def any_of(cls, values: list[AcceptedFileTypes]) -> FileType:
        return FileType(subject='file_type', matcher=AnyOf(values=values))
    
    @classmethod
    def all_of(cls, values: list[AcceptedFileTypes]) -> FileType:
        return FileType(subject='file_type', matcher=AllOf(values=values))
    
    @classmethod
    def none_of(cls, values: list[AcceptedFileTypes]) -> FileType:
        return FileType(subject='file_type', matcher=NoneOf(values=values))
    

class FolderPath(BaseSubjectFilter):
    subject: Literal['folder_path'] = 'folder_path'
    matcher: AnyOf[str] | NoneOf[str] | Prefix | Suffix
    
    @classmethod
    def any_of(cls, values: list[str]) -> FolderPath:
        return FolderPath(subject='folder_path', matcher=AnyOf(values=values))
    
    @classmethod
    def none_of(cls, values: list[str]) -> FolderPath:
        return FolderPath(subject='folder_path', matcher=NoneOf(values=values))
    
    @classmethod
    def prefix(cls, value: str) -> FolderPath:
        return FolderPath(subject='folder_path', matcher=Prefix(value=value))
    
    @classmethod
    def suffix(cls, value: str) -> FolderPath:
        return FolderPath(subject='folder_path', matcher=Suffix(value=value))
    
    
class ID(BaseSubjectFilter):
    subject: Literal['id'] = 'id'
    matcher: AnyOf[str] | NoneOf[str]
    
    @classmethod
    def any_of(cls, values: list[str]) -> ID:
        return ID(subject='id', matcher=AnyOf(values=values))
    
    @classmethod
    def none_of(cls, values: list[str]) -> ID:
        return ID(subject='id', matcher=NoneOf(values=values))
    

class Issue(BaseSubjectFilter):
    subject: Literal['issue'] = 'issue'
    matcher: AnyOf[IssueType] | NoneOf[IssueType]
    
    @classmethod
    def any_of(cls, values: list[IssueType]) -> Issue:
        return Issue(subject='issue', matcher=AnyOf(values=values))
    
    @classmethod
    def none_of(cls, values: list[IssueType]) -> Issue:
        return Issue(subject='issue', matcher=NoneOf(values=values))
    

class ItemName(BaseSubjectFilter):
    subject: Literal['item_name'] = 'item_name'
    matcher: AnyOf[str] | NoneOf[str] | Prefix | Suffix | Contains | NotContains
    
    @classmethod
    def any_of(cls, values: list[str]) -> ItemName:
        return ItemName(subject='item_name', matcher=AnyOf(values=values))
    
    @classmethod
    def none_of(cls, values: list[str]) -> ItemName:
        return ItemName(subject='item_name', matcher=NoneOf(values=values))
    
    @classmethod
    def prefix(cls, value: str) -> ItemName:
        return ItemName(subject='item_name', matcher=Prefix(value=value))
    
    @classmethod
    def suffix(cls, value: str) -> ItemName:
        return ItemName(subject='item_name', matcher=Suffix(value=value))
    
    @classmethod
    def contains(cls, value: str) -> ItemName:
        return ItemName(subject='item_name', matcher=Contains(value=value))
    
    @classmethod
    def not_contains(cls, value: str) -> ItemName:
        return ItemName(subject='item_name', matcher=NotContains(value=value))


class ProcessingStatus(BaseSubjectFilter):
    subject: Literal['processing_status'] = 'processing_status'
    matcher: AnyOf[ProcessingStatusType] | NoneOf[ProcessingStatusType]
    
    @classmethod
    def any_of(cls, values: list[ProcessingStatusType]) -> ProcessingStatus:
        return ProcessingStatus(subject='processing_status', matcher=AnyOf(values=values))
    
    @classmethod
    def none_of(cls, values: list[ProcessingStatusType]) -> ProcessingStatus:
        return ProcessingStatus(subject='processing_status', matcher=NoneOf(values=values))
    
    
class UpdatedAt(BaseSubjectFilter):
    subject: Literal['updated_at'] = 'updated_at'
    matcher: DateRange
    
    @classmethod
    def between(cls, start: datetime, end: datetime) -> UpdatedAt:
        return UpdatedAt(subject='updated_at', matcher=DateRange(start=start, end=end))
    
    @classmethod
    def before(cls, end: datetime) -> UpdatedAt:
        return UpdatedAt(subject='updated_at', matcher=DateRange(end=end))
    
    @classmethod
    def after(cls, start: datetime) -> UpdatedAt:
        return UpdatedAt(subject='updated_at', matcher=DateRange(start=start))
    
    
    
class WorkflowStatus(BaseSubjectFilter):
    subject: Literal['workflow_status'] = 'workflow_status'
    matcher: AnyOf[WorkflowStatusType] | NoneOf[WorkflowStatusType]
    
    @classmethod
    def any_of(cls, values: list[WorkflowStatusType]) -> WorkflowStatus:
        return WorkflowStatus(subject='workflow_status', matcher=AnyOf(values=values))
    
    @classmethod
    def none_of(cls, values: list[WorkflowStatusType]) -> WorkflowStatus:
        return WorkflowStatus(subject='workflow_status', matcher=NoneOf(values=values))
    
    
class WorkflowStage(BaseSubjectFilter):
    subject: Literal['workflow_stage'] = 'workflow_stage'
    matcher: AnyOf[str] | NoneOf[str]
    
    @classmethod
    def any_of(cls, values: list[str]) -> WorkflowStage:
        return WorkflowStage(subject='workflow_stage', matcher=AnyOf(values=values))
    
    @classmethod
    def none_of(cls, values: list[str]) -> WorkflowStage:
        return WorkflowStage(subject='workflow_stage', matcher=NoneOf(values=values))
    

# Matchers
class AnyOf(BaseMatcher, Generic[T]):
    name: Literal['any_of'] = 'any_of'
    values: List[T]
    
    @validator('values')
    def validate_any_of(cls, value: List[T]) -> List[T]:
        if len(value) < 2:
            raise ValueError("Must provide at least two values for 'any_of' matcher.")
        return value
    
class AllOf(BaseMatcher, Generic[T]):
    name: Literal['all_of'] = 'all_of'
    values: List[T]
    
    @validator('values')
    def validate_all_of(cls, value: List[T]) -> List[T]:
        if len(value) < 1:
            raise ValueError("Must provide at least a value for 'all_of' matcher.")
        return value
    
class NoneOf(BaseMatcher, Generic[T]):
    name: Literal['none_of'] = 'none_of'
    values: List[T]
    
    @validator('values')
    def validate_none_of(cls, value: List[T]) -> List[T]:
        if len(value) < 1:
            raise ValueError("Must provide at least a value for 'none_of' matcher.")
        return value

class Equals(BaseMatcher, Generic[T]):
    name: Literal['equals'] = 'equals'
    value: T
    
class DateRange(BaseModel):
    name: str = 'date_range'
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    @validator('start', 'end')
    def validate_date_range(cls, value: Optional[datetime], values: dict) -> Optional[datetime]:
        if not values.get('start') and not values.get('end'):
            raise ValueError("At least one of 'start' or 'end' must be provided.")
        return value
    
class Prefix(BaseMatcher):
    name: Literal['prefix'] = 'prefix'
    value: str

class Suffix(BaseMatcher):
    name: Literal['suffix'] = 'suffix'
    value: str

class Contains(BaseMatcher):
    name: Literal['contains'] = 'contains'
    value: str

class NotContains(BaseMatcher):
    name: Literal['not_contains'] = 'not_contains'
    value: str
    
