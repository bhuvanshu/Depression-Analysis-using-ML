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

    @Column(nullable = false)
    private Double probabilityScore;

    @Column(nullable = false)
    private String riskLevel;

    @Column(length = 500)
    private String recommendation;

    private LocalDateTime predictedAt = LocalDateTime.now();

    @OneToOne
    @JoinColumn(name = "response_id")
    private ScreeningResponse screeningResponse;
}