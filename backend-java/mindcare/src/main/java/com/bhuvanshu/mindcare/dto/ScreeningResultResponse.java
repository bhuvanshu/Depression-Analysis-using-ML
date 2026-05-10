package com.bhuvanshu.mindcare.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ScreeningResultResponse {

    private Integer prediction;

    @JsonProperty("prediction_label")
    private String prediction_label;

    private Probability probability;

    @JsonProperty("risk_level")
    private String risk_level;

    @JsonProperty("recommended_action")
    private String recommended_action;

    @JsonProperty("risk_percentile")
    private String risk_percentile;

    private String status;
}