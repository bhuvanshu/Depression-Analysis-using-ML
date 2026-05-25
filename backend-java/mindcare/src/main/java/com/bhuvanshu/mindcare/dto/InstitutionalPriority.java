package com.bhuvanshu.mindcare.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class InstitutionalPriority {
    private String tier;
    
    @JsonProperty("percentile_group")
    private String percentileGroup;
    
    private String action;
    private String color;
}
