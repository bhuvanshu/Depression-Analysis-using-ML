package com.bhuvanshu.mindcare.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ScreeningRequest {

    @NotBlank(message = "Enrollment ID is required")
    private String enrollmentId;

    @NotNull(message = "Age is required")
    @Min(value = 10, message = "Age must be at least 10")
    @Max(value = 60, message = "Age must be at most 60")
    private Integer age;

    @JsonProperty("academic_pressure")
    @NotNull(message = "Academic pressure is required")
    @Min(value = 0, message = "Academic pressure must be between 0 and 5")
    @Max(value = 5, message = "Academic pressure must be between 0 and 5")
    private Integer academicPressure;

    @Min(value = 0, message = "CGPA must be between 0 and 10")
    @Max(value = 10, message = "CGPA must be between 0 and 10")
    private Float cgpa;

    @JsonProperty("study_satisfaction")
    @NotNull(message = "Study satisfaction is required")
    @Min(value = 0, message = "Study satisfaction must be between 0 and 5")
    @Max(value = 5, message = "Study satisfaction must be between 0 and 5")
    private Integer studySatisfaction;

    @JsonProperty("work_study_hours")
    @NotNull(message = "Work/study hours is required")
    @Min(value = 0, message = "Work/study hours must be between 0 and 5")
    @Max(value = 5, message = "Work/study hours must be between 0 and 5")
    private Integer workStudyHours;

    @JsonProperty("financial_stress")
    @NotNull(message = "Financial stress is required")
    @Min(value = 0, message = "Financial stress must be between 0 and 5")
    @Max(value = 5, message = "Financial stress must be between 0 and 5")
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
