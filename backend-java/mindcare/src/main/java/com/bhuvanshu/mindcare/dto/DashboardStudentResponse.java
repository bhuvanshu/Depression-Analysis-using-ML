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

    private Double probabilityScore;

    // Questionnaire metrics

    private Integer academicPressure;

    private Integer financialStress;

    private Integer studySatisfaction;

    private Boolean suicidalThoughts;

    private Boolean familyHistory;

    private Integer studyHours;

    private String sleepDuration;

    // Screening timestamp

    private LocalDateTime screeningDate;
}