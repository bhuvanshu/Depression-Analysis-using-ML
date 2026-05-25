package com.bhuvanshu.mindcare.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ScreeningResultResponse {

    private Integer prediction;

    @JsonProperty("prediction_label")
    private String predictionLabel;

    private Probability probability;

    @JsonProperty("risk_level")
    private String riskLevel;

    @JsonProperty("recommended_action")
    private String recommendedAction;

    @JsonProperty("risk_percentile")
    private String riskPercentile;

    private String status;

    @JsonProperty("severity_interpretation")
    private SeverityInterpretation severityInterpretation;

    @JsonProperty("institutional_priority")
    private InstitutionalPriority institutionalPriority;
}