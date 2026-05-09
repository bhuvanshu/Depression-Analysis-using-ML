package com.bhuvanshu.mindcare.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ScreeningRequest {

    private String enrollmentId;

    private Integer age;

    private Integer academic_pressure;

    private Float cgpa;

    private Integer study_satisfaction;

    private Integer work_study_hours;

    private Integer financial_stress;

    private Boolean suicidal_thoughts;

    private Boolean family_history;

    private String gender;

    private String sleep_duration;

    private String degree;
}