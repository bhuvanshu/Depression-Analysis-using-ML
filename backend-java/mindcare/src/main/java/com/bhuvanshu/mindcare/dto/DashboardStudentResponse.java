package com.bhuvanshu.mindcare.dto;

import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
public class DashboardStudentResponse {

    private String enrollmentId;

    private String studentName;

    private String department;

    private String riskLevel;

    private String severityLevel;

    private Double probabilityScore;

    private String recommendation;

    // Questionnaire metrics

    private Integer academicPressure;

    private Integer financialStress;

    private Integer studySatisfaction;

    private Boolean suicidalThoughts;

    private Boolean familyHistory;

    private Integer studyHours;

    private String sleepDuration;

    private Float cgpa;

    // Screening timestamp

    private LocalDateTime screeningDate;
}
