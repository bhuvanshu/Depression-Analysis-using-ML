package com.bhuvanshu.mindcare.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "screening_responses")
public class ScreeningResponse {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long responseId;

    private Integer academicPressure;

    private Integer financialStress;

    private Integer studySatisfaction;

    private Integer sleepDuration;

    private Integer workStudyHours;

    private Boolean suicidalThoughts;

    private Boolean familyHistory;

    private Float cgpa;

    private LocalDateTime submittedAt = LocalDateTime.now();

    @ManyToOne
    @JoinColumn(name = "student_id")
    private Student student;

    // Getters and Setters
}