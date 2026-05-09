package com.bhuvanshu.mindcare.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class DashboardStudentResponse {

    private String enrollmentId;

    private String studentName;

    private String department;

    private String riskLevel;

    private Double probabilityScore;
}