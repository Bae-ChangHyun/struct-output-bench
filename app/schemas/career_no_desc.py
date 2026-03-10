"""description이 없는 Career 스키마"""
from pydantic import BaseModel
from typing import List, Optional, Literal


class CareerNoDesc(BaseModel):
    company_name: Optional[str] = None
    is_company_private: Optional[bool] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    is_currently_employed: Optional[bool] = None
    department: Optional[str] = None
    position: Optional[str] = None
    work_details: Optional[str] = None
    annual_salary: Optional[str] = None
    reason_for_leaving: Optional[str] = None
    employment_type: Optional[str] = None
    work_location: Optional[str] = None


class ActivityExperienceNoDesc(BaseModel):
    activity_type: Optional[str] = None
    activity_name: Optional[str] = None
    organization: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    details: Optional[str] = None


class OverseasExperienceNoDesc(BaseModel):
    experience_type: Optional[Literal["어학연수", "교환학생", "워킹홀리데이", "유학"]] = None
    country: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    details: Optional[str] = None


class LanguageSkillNoDesc(BaseModel):
    assessment_type: Optional[Literal["회화능력", "공인시험"]] = None
    language: Optional[str] = None
    proficiency_level: Optional[str] = None
    test_name: Optional[str] = None
    test_language: Optional[str] = None
    test_score: Optional[str] = None
    test_date: Optional[str] = None


class CertificateNoDesc(BaseModel):
    certificate_name: Optional[str] = None
    issuing_authority: Optional[str] = None
    acquisition_date: Optional[str] = None


class AwardExperienceNoDesc(BaseModel):
    award_name: Optional[str] = None
    organizer: Optional[str] = None
    award_date: Optional[str] = None
    details: Optional[str] = None


class EmploymentAndMilitaryInfoNoDesc(BaseModel):
    is_veteran_target: Optional[bool] = None
    veteran_reason: Optional[str] = None
    is_employment_protection_target: Optional[bool] = None
    is_disabled: Optional[bool] = None
    disability_grade: Optional[str] = None
    military_status: Optional[Literal["군필", "미필", "면제", "해당없음"]] = None
    service_start_date: Optional[str] = None
    service_end_date: Optional[str] = None
    military_branch: Optional[str] = None
    rank: Optional[str] = None


class OnlineProfileNoDesc(BaseModel):
    sns_links: List[str] = []


class MainInfoNoDesc(BaseModel):
    careers: List[CareerNoDesc] = []
    activity_experiences: List[ActivityExperienceNoDesc] = []
    overseas_experiences: List[OverseasExperienceNoDesc] = []
    language_skills: List[LanguageSkillNoDesc] = []
    certificates: List[CertificateNoDesc] = []
    award_experiences: List[AwardExperienceNoDesc] = []
    employment_military_info: Optional[EmploymentAndMilitaryInfoNoDesc] = None
    sns: Optional[OnlineProfileNoDesc] = None
