from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union

# 경력
class Career(BaseModel):
    company_name: Optional[str] = Field(
        title="기업명",
        description="기업명",
        default=None
    )
    is_company_private: Optional[bool] = Field(
        title="기업명 비공개 여부",
        description="기업명 비공개 여부(공개 또는 비공개)",
        default=None
    )
    start_date: Optional[str] = Field(
        title="입사날짜",
        description="입사 날짜",
        format="date",
        pattern=r"^\d{4}(-\d{2})?(-\d{2})?$",  
        default=None
    )
    end_date: Optional[str] = Field(
        title="퇴사날짜",
        description="퇴사 날짜",
        format="date",
        pattern=r"^\d{4}(-\d{2})?(-\d{2})?$",  
        default=None
    )
    is_currently_employed: Optional[bool] = Field(
        title="재직중 여부",
        description="재직중 여부",
        default=None
    )
    department: Optional[str] = Field(
        title="부서명",
        description="부서명",
        default=None
    )
    position: Optional[str] = Field(
        title="직책/직급",
        description="직책/직급(사원/대리/과장/차장 등)",
        default=None
    )

    # 업무 정보
    work_details: Optional[str] = Field(
        title="담당업무",
        description="담당업무(해당 회사에서의 경력요약, 경력기술서, 주요업무, 담당업무, 프로젝트 상세 내용 등 모든 경력 관련 전체 내용)",
        default=None
    )
    annual_salary: Optional[str] = Field(
        title="해당 기업 연봉",
        description="해당 기업 연봉",
        default=None
    )
    reason_for_leaving: Optional[str] = Field(
        title="이직사유",
        description="이직사유",
        default=None
    )
    employment_type: Optional[str] = Field(
        title="고용형태",
        description="고용형태(정규직/계약직/파견직 등)",
        default=None
    )
    work_location: Optional[str] = Field(
        title="근무지역",
        description="근무지역",
        default=None
    )


# 활동/경험
class ActivityExperience(BaseModel):
    activity_type: Optional[str] = Field(
        title="활동구분",
        description="활동구분(인턴/대외활동/교육프로그램/봉사/부트캠프 등)",
        default=None
    )
    activity_name: Optional[str] = Field(
        title="활동명",
        description="활동명",
        default=None
    )
    organization: Optional[str] = Field(
        title="기관/회사/교육기관",
        description="기관/회사/교육기관",
        default=None
    )
    start_date: Optional[str] = Field(
        title="시작날짜",
        description="시작 날짜",
        format="date",
        pattern=r"^\d{4}(-\d{2})?(-\d{2})?$",  
        default=None
    )
    end_date: Optional[str] = Field(
        title="종료날짜",
        description="종료 날짜",
        format="date",
        pattern=r"^\d{4}(-\d{2})?(-\d{2})?$",  
        default=None
    )
    details: Optional[str] = Field(
        title="활동상세내용",
        description="활동상세내용",
        default=None
    )


# 해외경험
class OverseasExperience(BaseModel):
    experience_type: Optional[Literal["어학연수", "교환학생", "워킹홀리데이", "유학"]] = Field(
        title="해외경험 유형",
        description="해외경험 유형",
        default=None
    )
    country: Optional[str] = Field(
        title="국가명",
        description="국가명",
        default=None
    )
    start_date: Optional[str] = Field(
        title="시작날짜",
        description="시작 날짜",
        format="date",
        pattern=r"^\d{4}(-\d{2})?(-\d{2})?$",  
        default=None
    )
    end_date: Optional[str] = Field(
        title="종료날짜",
        description="종료 날짜",
        format="date",
        pattern=r"^\d{4}(-\d{2})?(-\d{2})?$",  
        default=None
    )
    details: Optional[str] = Field(
        title="해외경험 상세내용",
        description="해외경험 상세내용",
        default=None
    )


# 어학능력
class LanguageSkill(BaseModel):
    assessment_type: Optional[Literal["회화능력", "공인시험"]] = Field(
        title="어학구분",
        description="어학구분(회화능력/공인시험)",
        default=None
    )
    language: Optional[str] = Field(
        title="회화 가능 언어",
        description="회화 가능 언어(영어/한국어/중국어/일본어 등)",
        default=None
    )
    proficiency_level: Optional[str] = Field(
        title="회화 수준",
        description="회화 수준(일상회화/비즈니스회화/원어민급)",
        default=None
    )
    test_name: Optional[str] = Field(
        title="공인 시험명",
        description="공인 시험명(TOEIC/TOEFL/HSK/JLPT 등)",
        default=None
    )
    test_language: Optional[str] = Field(
        title="공인 시험 외국어명",
        description="공인 시험 외국어명(영어/중국어/일본어/한국어 등)",
        default=None
    )
    test_score: Optional[str] = Field(
        title="공인 시험 점수",
        description="공인 시험 점수",
        default=None
    )
    test_date: Optional[str] = Field(
        title="공인 시험 취득날짜",
        description="공인 시험 취득 날짜",
        format="date",
        pattern=r"^\d{4}(-\d{2})?(-\d{2})?$",  
        default=None
    )


# 자격증
class Certificate(BaseModel):
    certificate_name: Optional[str] = Field(
        title="자격&면허명",
        description="자격&면허명",
        default=None
    )
    issuing_authority: Optional[str] = Field(
        title="발행처/수여기관",
        description="발행처/수여기관",
        default=None
    )
    acquisition_date: Optional[str] = Field(
        title="발행날짜",
        description="발행날짜",
        format="date",
        pattern=r"^\d{4}(-\d{2})?(-\d{2})?$",  
        default=None
    )


# 수상 경력
class AwardExperience(BaseModel):
    award_name: Optional[str] = Field(
        title="수상명/상장명",
        description="수상명/상장명",
        default=None
    )
    organizer: Optional[str] = Field(
        title="주최기관",
        description="주최기관",
        default=None
    )
    award_date: Optional[str] = Field(
        title="수상날짜",
        description="수상날짜",
        format="date",
        pattern=r"^\d{4}(-\d{2})?(-\d{2})?$",  
        default=None
    )
    details: Optional[str] = Field(
        title="수상 상세내용",
        description="수상 상세내용",
        default=None
    )


# 취업우대/병역
class EmploymentAndMilitaryInfo(BaseModel):
    # 취업우대
    is_veteran_target: Optional[bool] = Field(
        title="보훈대상여부",
        description="보훈대상여부",
        default=None
    )
    veteran_reason: Optional[str] = Field(
        title="보훈사유",
        description="보훈사유",
        default=None
    )
    is_employment_protection_target: Optional[bool] = Field(
        title="취업보호대상여부",
        description="취업보호대상여부",
        default=None
    )
    is_disabled: Optional[bool] = Field(
        title="장애여부",
        description="장애여부",
        default=None
    )
    disability_grade: Optional[str] = Field(
        title="장애등급",
        description="장애등급",
        default=None
    )
    
    # 병역
    military_status: Optional[Literal["군필", "미필", "면제", "해당없음"]] = Field(
        title="병역대상",
        description="병역대상(군필/미필/면제/해당없음)",
        default=None
    )
    service_start_date: Optional[str] = Field(
        title="입대날짜",
        description="입대 날짜",
        format="date",
        pattern=r"^\d{4}(-\d{2})?(-\d{2})?$",  
        default=None
    )
    service_end_date: Optional[str] = Field(
        title="제대날짜",
        description="제대 날짜",
        format="date",
        pattern=r"^\d{4}(-\d{2})?(-\d{2})?$",  
        default=None
    )
    military_branch: Optional[str] = Field(
        title="군별",
        description="군별(육군/해군/공군/해병 등)",
        default=None
    )
    rank: Optional[str] = Field(
        title="제대계급",
        description="제대계급(이등병/일병/상병/병장 등)",
        default=None
    )


# 포트폴리오/SNS
class OnlineProfile(BaseModel):
    sns_links: List[str] = Field(
        title="SNS 링크",
        description="SNS 링크(깃헙/링크드인/블로그 등)",
        minitems=0,  
        maxitems=10,  
        default_factory=list
    )
    
class MainInfo(BaseModel):
    careers: List[Career] = Field(
        title="경력",
        description="경력",
        minitems=0,
        maxitems=20,
        default_factory=list
    )
    activity_experiences: List[ActivityExperience] = Field(
        title="활동/경험",
        description="활동/경험(대외활동/교육프로그램/봉사/부트캠프 등)",
        minitems=0,
        maxitems=15,
        default_factory=list
    )
    overseas_experiences: List[OverseasExperience] = Field(
        title="해외경험",
        description="해외경험",
        minitems=0,
        maxitems=10,
        default_factory=list
    )
    language_skills: List[LanguageSkill] = Field(
        title="어학능력",
        description="어학능력",
        minitems=0,
        maxitems=10,
        default_factory=list
    )
    certificates: List[Certificate] = Field(
        title="자격/면허",
        description="자격/면허",
        minitems=0,
        maxitems=10,
        default_factory=list
    )
    award_experiences: List[AwardExperience] = Field(
        title="수상 경력",
        description="수상 경력",
        minitems=0,
        maxitems=10,
        default_factory=list
    )
    employment_military_info: Optional[EmploymentAndMilitaryInfo] = Field(
        title="취업우대/병역사항",
        description="취업우대/병역사항",
        default=None
    )
    sns: Optional[OnlineProfile] = Field(
        title="SNS 링크",
        description="SNS 링크",
        default=None
    )