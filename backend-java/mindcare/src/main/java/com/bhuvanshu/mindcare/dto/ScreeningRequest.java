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
    private Integer academic_pressure;

    private Float cgpa;

    @JsonProperty("study_satisfaction")
    private Integer study_satisfaction;

    @JsonProperty("work_study_hours")
    private Integer work_study_hours;

    @JsonProperty("financial_stress")
    private Integer financial_stress;

    @JsonProperty("suicidal_thoughts")
    private Boolean suicidal_thoughts;

    @JsonProperty("family_history")
    private Boolean family_history;

    private String gender;

    @JsonProperty("sleep_duration")
    private String sleep_duration;

    private String degree;
}