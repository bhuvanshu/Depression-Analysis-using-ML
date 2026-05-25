package com.bhuvanshu.mindcare.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ScreeningRequest {

    private String enrollmentId;

    private Integer age;

    @JsonProperty("academic_pressure")
    private Integer academicPressure;

    private Float cgpa;

    @JsonProperty("study_satisfaction")
    private Integer studySatisfaction;

    @JsonProperty("work_study_hours")
    private Integer workStudyHours;

    @JsonProperty("financial_stress")
    private Integer financialStress;

    @JsonProperty("suicidal_thoughts")
    private Boolean suicidalThoughts;

    @JsonProperty("family_history")
    private Boolean familyHistory;

    private String gender;

    @JsonProperty("sleep_duration")
    private String sleepDuration;

    private String degree;
}