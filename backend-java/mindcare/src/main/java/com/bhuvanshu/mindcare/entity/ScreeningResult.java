package com.bhuvanshu.mindcare.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@Table(name = "screening_results")
public class ScreeningResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long resultId;

    private Double probabilityScore;

    private String riskLevel;

    private String recommendation;

    private LocalDateTime predictedAt = LocalDateTime.now();

    @OneToOne
    @JoinColumn(name = "response_id")
    private ScreeningResponse screeningResponse;
}