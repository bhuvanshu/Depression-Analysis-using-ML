package com.bhuvanshu.mindcare.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ScreeningResultResponse {

    private Integer prediction;

    private String prediction_label;

    private Probability probability;

    private String risk_level;

    private String recommended_action;

    private String risk_percentile;

    private String status;
}